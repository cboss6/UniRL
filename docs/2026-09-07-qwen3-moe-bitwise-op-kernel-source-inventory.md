# Qwen3-MoE bitwise 对齐 op/kernel 来源清单

日期：2026-09-07

## 1. 范围

本文统计当前 `MyUniRL4` + `UniMatch` 工作区中，Qwen3-30B-A3B
rollout-vs-old-logprob 严格门禁实际依赖的 op/kernel 来源。

覆盖两条已验证路径：

- FSDP actor world=4 / vLLM rollout TP4、EP1
- VeOmni actor EP4 / 四个 vLLM rollout TP1、EP1

old logprob 由 actor 对固定 `prompt+response` 执行整段 no-grad exact
forward 得到。当前验收条件为 FP32 `max_absdiff=0`、K3 mean/max=0、
`torch.equal=True`。

本文只把该 no-grad old-policy forward 与 rollout 热路径列为
“bitwise 签核路径”。承载梯度的 forward/backward 在第 8 节单列，不能与
rollout-vs-old 门禁混为一谈。

## 2. 来源标签

- **UM-CUSTOM**：UniMatch 自研的 CuTe/CUTLASS/Triton kernel 或自定义
  `torch.autograd.Function`。
- **VLLM-BI-DIRECT**：运行时直接 import 并调用 vLLM 安装包中的
  batch-invariant primitive。
- **VLLM-BI-COPY**：算法和 kernel 来自 vLLM batch-invariant 实现，但源码
  复制/去依赖后保存在 UniMatch 中。
- **VLLM-NATIVE**：vLLM 自带的普通模型层、FA3、MoE permute、TP group
  等；不是 UniMatch 自研 kernel。
- **TORCH**：PyTorch/ATen eager op、Transformers eager expression、
  `torch.distributed` 或 FSDP。
- **EXTERNAL**：既不属于上述三者的独立依赖，例如 DeepEP、
  `vllm_softmax_kernel`、NCCL/CUDA runtime。
- **UM-WRAPPER**：UniMatch 只负责接线、固定参数或切换 vLLM/Torch
  primitive；底层计算 kernel 仍按真实来源归类。

“源码位于 UniMatch”不自动等于 UM-CUSTOM。例如 log-softmax 文件位于
UniMatch，但它是 vLLM batch-invariant reduction 的自包含副本，应归为
VLLM-BI-COPY。

## 3. 当前启用开关

两条 launcher 都启用：

```text
VLLM_BATCH_INVARIANT=0
UNIMATCH_PATCHES=
  qwen3_moe,
  batch_invariant_reductions,
  batch_invariant_norm,
  batch_invariant_precision,
  batch_invariant_attention
UNIMATCH_QWEN3_MOE_COMPONENTS=
  experts_grouped,
  qkv_local,
  oproj_col,
  lm_head_col,
  gate_fp32,
  router_hf
UNIMATCH_HF_ATEN_PATCHES=linear,softmax
UNIMATCH_ROUTER_SOFTMAX_IMPL=vllm_kernel
UNIMATCH_ATTENTION_BACKEND=fa3
UNIMATCH_MEGATRON=0
UNIMATCH_MOE_SUM=bf16
```

FSDP/TP4：

```text
UNIMATCH_TRAIN_TP=4
UNIMATCH_TRAIN_EP_SIZE=1
```

VeOmni EP4 / vLLM TP1：

```text
UNIMATCH_VEOMNI_EP4_EXACT=1
UNIMATCH_TRAIN_TP=1
UNIMATCH_TRAIN_EP_SIZE=4
UNIMATCH_GROUPED_VARIANT=tma
UNIMATCH_DENSE_VARIANT=tma
UNIMATCH_DEEPEP_MODE=ht
UNIMATCH_DEEPEP_ASYNC_FINISH=0
```

入口：

- `examples/run_qwen3_moe_fsdp_vllm_tp4_bitwise.sh:26-45`
- `examples/run_qwen3_moe_veomni_ep4_vllm_tp1_bitwise.sh:36-66`
- UniMatch vLLM plugin：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/__init__.py:42-86`

## 4. 来源统计摘要

按“唯一 kernel/primitive 家族”而不是调用次数统计：

### 4.1 UniMatch 自研 device kernel 家族

1. CuTe DSL RMSNorm forward/backward
2. BF16-input/FP32-output router gate GEMM
3. TP column-parallel dense GEMM/GEMV + fused AllGather
4. TP column-parallel grouped GEMM/GEMV + fused AllGather
5. VeOmni rank-local fused grouped FC1/FC2
6. EP-ordered MoE combine

### 4.2 vLLM batch-invariant

- **直接调用 1 个核心家族**：`linear_batch_invariant`
- **UniMatch 内自包含副本 3 个 reduction 家族**：
  log-softmax、softmax、mean
- **仅保留未启用副本**：vLLM batch-invariant Triton RMSNorm 副本存在于
  `norm_triton.py`，但当前热路径实际使用 UniMatch CuTe RMSNorm

### 4.3 vLLM 原生 kernel/primitive

- `VocabParallelEmbedding`
- Hopper FA3 / `vllm_flash_attn`
- `moe_permute`
- vLLM TP group AllGather 与模型层 orchestration

### 4.4 Torch/ATen eager 逻辑

- HF embedding
- RoPE 频率构造及显式乘加/拼接
- `torch.topk`
- stable sort、`bincount`、gather/scatter/indexing
- FSDP actor 的 eager SwiGLU
- routing weight 乘法及 VeOmni rank-local partial 的显式 FP32 slot fold

### 4.5 外部依赖

- `vllm_softmax_kernel.row_softmax`
- DeepEP HT `Buffer.dispatch/combine`
- NCCL/CUDA/PyTorch distributed collectives

## 5. 公共 Transformer 与输出层清单

| op/stage | vLLM rollout TP4 | vLLM rollout TP1（EP4 配置） | FSDP old-policy actor | VeOmni EP4 old-policy actor | 主来源 |
|---|---|---|---|---|---|
| Token embedding | vLLM `VocabParallelEmbedding`，TP sum | vLLM `VocabParallelEmbedding`，TP1 | Transformers/HF embedding | VeOmni 生成模型的 HF embedding | VLLM-NATIVE / TORCH |
| Residual add | stock vLLM eager/fused orchestration | 同左 | PyTorch BF16 add | VeOmni/PyTorch BF16 add | VLLM-NATIVE / TORCH |
| Decoder/final RMSNorm | vLLM `RMSNorm.forward_cuda` 被替换为 `unimatch.ops.rmsnorm.rms_norm_fwd` | 同左 | `Qwen3MoeRMSNorm.forward -> rmsnorm_bi` | VeOmni OpSlot `unimatch_bi` | UM-CUSTOM |
| QKV projection | UniMatch local-only TP column-parallel dense GEMM，固定 full-K 累加 | TP1 特例直接调用 `linear_batch_invariant` | ATen linear exact override 直接调用 `linear_batch_invariant` | ATen linear exact override直接调用 `linear_batch_invariant` | UM-CUSTOM（TP4 rollout）/ VLLM-BI-DIRECT（actors、TP1） |
| Q/K RMSNorm | 同 decoder RMSNorm | 同 decoder RMSNorm | 同 decoder RMSNorm | 同 decoder RMSNorm | UM-CUSTOM |
| RoPE frequency | UniMatch canonical FP32 inverse-frequency patch | 同左 | UniMatch FSDP patch，以 Torch FP32 构造 | UniMatch `rotary_frequencies_vllm_exact`，Torch FP32 | UM-WRAPPER + TORCH |
| RoPE apply | UniMatch 替换 vLLM CUDA RoPE 为显式 eager NeoX 运算 | 同左 | Transformers eager rotate | VeOmni OpSlot `unimatch_vllm_exact`，内部为 Torch BF16 乘加/拼接 | UM-WRAPPER + TORCH |
| Attention core | vLLM Hopper FA3；UniMatch 仅激活 vLLM attention BI gate，强制 `num_splits=1` | 同左 | UniMatch wrapper 调用 vLLM bundled FA3 varlen，`deterministic=True, num_splits=1` | 同一 vLLM bundled FA3 contract | VLLM-NATIVE + UM-WRAPPER |
| Attention o_proj | UniMatch TP column-parallel dense + input AllGather + fused output AllGather | TP1 特例直接调用 `linear_batch_invariant` | exact ATen linear -> `linear_batch_invariant` | exact ATen linear -> `linear_batch_invariant` | UM-CUSTOM（TP4 rollout）/ VLLM-BI-DIRECT（actors、TP1） |
| Router gate GEMM | UniMatch `DenseGemmBF16Fp32Out` | 同左 | exact ATen 特例 `(N=128,K=2048)` -> UniMatch BF16/FP32-out GEMM | 相同 ATen 特例 | UM-CUSTOM |
| Router softmax | `vllm_softmax_kernel.row_softmax` | 同左 | ATen softmax override -> 同一 `row_softmax` | ATen softmax override -> 同一 `row_softmax` | EXTERNAL |
| Router top-k | `torch.topk` | `torch.topk` | `torch.topk` | `torch.topk` | TORCH |
| Router renormalize | Torch `sum` + division，HF 顺序 | 同左 | Torch `sum` + division | Torch `sum` + division | TORCH |
| LM head | UniMatch TP column-parallel dense + fused AllGather | TP1 特例直接调用 `linear_batch_invariant` | exact ATen linear -> `linear_batch_invariant` | exact ATen linear -> `linear_batch_invariant` | UM-CUSTOM（TP4 rollout）/ VLLM-BI-DIRECT（actors、TP1） |
| Selected-token log-softmax | UniMatch ATen/vLLM sampler patch调用 vLLM-BI reduction 副本 | 同左 | `reductions_triton.log_softmax(...).gather(...)` | 同左 | VLLM-BI-COPY + TORCH gather |
| FP32 old-vs-rollout max absdiff/K3 | 不在模型 kernel 内；UniRL 统一比较 | 同左 | `float()`、sub/abs/max、`expm1` | 同左 | TORCH |

关键源码：

- vLLM direct BI linear：
  `/opt/conda/envs/unirl/lib/python3.12/site-packages/vllm/model_executor/layers/batch_invariant.py:890-897`
- FSDP ATen linear 分派：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/fsdp/hf_aten.py:54-77`
- vLLM TP4 dense replacements：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/qwen3_dualcol/qwen3_modules.py:583-743`
- RMSNorm patch：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/batch_invariant_norm/patch_norm.py:34-83`
- RMSNorm actor wrapper：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/functional/rmsnorm.py:22-52`
- canonical vLLM RoPE patch：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/qwen3_moe_dualcol/rope_contract.py:7-75`
- VeOmni RoPE：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/functional/rotary.py:11-65`
- actor FA3 wrapper：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/fsdp/attention.py:97-119`
- attention BI gate activation：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/batch_invariant_attention/patch_attention.py:11-48`
- BF16/FP32-out router gate：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/qwen3_moe_dualcol/gate_fp32.py:50-105`
- router softmax/top-k：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/qwen3_moe_dualcol/hf_router.py:11-58`
- log-softmax/reduction 副本：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/batch_invariant_reductions/reductions_triton.py:1-16`

## 6. FSDP/TP4 MoE 清单

| MoE stage | vLLM TP4 rollout | FSDP no-grad old-policy actor | 主来源 |
|---|---|---|---|
| Token-slot expansion | stock/vLLM model orchestration | `hidden.unsqueeze/expand/reshape` | VLLM-NATIVE / TORCH |
| Expert dispatch/sort | vLLM native `moe_permute` | `torch.argsort(stable=True)` + `torch.bincount` + indexing | VLLM-NATIVE / TORCH |
| Per-expert padding | UniMatch Torch indexing adapter，pad 到 grouped kernel block | actor loop 不需要相同的设备 padding | UM-WRAPPER + TORCH |
| FC1 gate/up | UniMatch `TpColumnParallelGroupedAuto`，WGMMA GEMM 或 decode GEMV，带 fused AllGather | 稳定 expert/rank loop 中的 `F.linear`；exact ATen override最终调用 vLLM `linear_batch_invariant` | UM-CUSTOM / VLLM-BI-DIRECT |
| SwiGLU | `F.silu(gate) * up`，BF16 中间实体化 | `F.silu(gate) * up` | TORCH |
| FC2/down | UniMatch `TpColumnParallelGroupedAuto` + fused AllGather | expert/rank loop中的 `F.linear` -> `linear_batch_invariant` | UM-CUSTOM / VLLM-BI-DIRECT |
| Routing weight multiply | Torch multiply + BF16 cast | Torch multiply + BF16 cast | TORCH |
| Slot combine | UniMatch `MoeCombineEpOrdered(sim_ep_size=1)` | 同一个 UniMatch `MoeCombineEpOrdered` | UM-CUSTOM |

关键源码：

- vLLM grouped rollout：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/qwen3_moe_dualcol/grouped_runner.py:652-895`
- vLLM native permute 调用：
  `grouped_runner.py:692-756`
- actor stable dispatch / expert loop：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/functional/qwen3_moe.py:81-103,163-178,364-438`
- actor strict/gradient wrapper：
  `unimatch/functional/qwen3_moe.py:480-544`
- EP-ordered combine：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/ops/moe_combine_ep_ordered/host.py:97-138,207-278`

## 7. VeOmni EP4 / vLLM TP1 MoE 清单

| MoE stage | VeOmni EP4 actor | vLLM TP1 rollout 模拟侧 | 主来源 |
|---|---|---|---|
| Global dispatch layout | `DeepEP Buffer.get_dispatch_layout` | 无跨 rank dispatch | EXTERNAL DeepEP |
| Hidden/ID/weight transport | `DeepEP Buffer.dispatch`，BF16 hidden + FP32 routing weight | 本地完整 128 experts | EXTERNAL DeepEP |
| Local stable grouping | `torch.argsort(stable=True)`、`bincount` | vLLM `moe_permute` + UniMatch pad adapter | TORCH / VLLM-NATIVE |
| Local FC1 | UniMatch `TpColumnParallelGroupedFusedGated`，TMA `(64,128)`，融合 HF eager SwiGLU | UniMatch grouped TMA kernel | UM-CUSTOM |
| SwiGLU | 融合在 UniMatch FC1 gated epilogue | UniMatch runner 中显式 Torch BF16 `F.silu * up` | UM-CUSTOM / TORCH |
| Local FC2 + routing weight | UniMatch `TpColumnParallelGroupedFusedElementwise(fusion="mul")` | UniMatch grouped FC2 后 Torch multiply + BF16 cast | UM-CUSTOM / TORCH |
| Rank-local partial | 按原 top-k slot 顺序显式 FP32 fold，末端 BF16 | `MoeCombineEpOrdered` 内模拟相同 rank bracket | TORCH（代码位于 UniMatch）/ UM-CUSTOM |
| EP combine transport | `DeepEP Buffer.combine` | 无真实 EP 通信 | EXTERNAL DeepEP |
| EP4 bracketed final sum | 真实 DeepEP rank partial + combine | `MoeCombineEpOrdered(sim_ep_size=4)` | EXTERNAL DeepEP / UM-CUSTOM |
| Dispatch/combine backward | UniMatch autograd wrapper调用 DeepEP reverse handle；本地专家 backward 由 VeOmni `EPMergedFc1GroupGemm` 重算 | rollout 无 backward | UM-WRAPPER + EXTERNAL DeepEP/VeOmni |

关键源码：

- VeOmni OpSlot 注册：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/veomni/bootstrap.py:23-39,320-374`
- DeepEP Buffer 创建和调用：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/veomni/dispatcher.py:59-82,273-327`
- VeOmni local grouped FC1/FC2：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/veomni/local_grouped.py:48-110,117-202`
- rank-local grouping/partial：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/veomni/grouped_experts.py:14-41,330-347`
- DeepEP adapter orchestration：
  `unimatch/adaptor/veomni/grouped_experts.py:356-415`
- TP1 rollout EP4 combine 模拟：
  `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/qwen3_moe_dualcol/grouped_runner.py:270-291,820-895`

## 8. 梯度路径（不属于 rollout-vs-old bitwise 门禁）

为了避免把“前向 bitwise 来源”误解成“全部训练 kernel 都 bitwise”，当前
gradient path 单列如下：

| 组件 | grad-enabled forward/backward | 来源 |
|---|---|---|
| Dense linear | exact mode 关闭时 `_native_linear -> torch.matmul`；backward 也使用 Torch matmul/sum | TORCH/cuBLAS |
| Log-softmax | 可微 Torch expression / checkpoint 路径 | TORCH |
| Attention forward | UniMatch autograd wrapper的 forward仍调用 vLLM FA3 | VLLM-NATIVE + UM-WRAPPER |
| Attention backward | `torch.nn.functional.scaled_dot_product_attention`，强制 `SDPBackend.MATH` 重算 | TORCH |
| FSDP MoE forward | UniMatch strict forward | UM-CUSTOM + VLLM-BI-DIRECT + TORCH |
| FSDP MoE backward | UniMatch autograd wrapper以可微 Torch expert graph重算 | UM-WRAPPER + TORCH |
| VeOmni MoE forward | UniMatch fused local grouped + DeepEP | UM-CUSTOM + EXTERNAL |
| VeOmni MoE backward | DeepEP reverse dispatch/combine + VeOmni `EPMergedFc1GroupGemm` 重算 | EXTERNAL DeepEP/VeOmni |
| RMSNorm backward | UniMatch CuTe DSL RMSNorm backward | UM-CUSTOM |

因此当前 `train|Δlogp|mean` 可以非零；它不改变第 1 节
rollout-vs-no-grad-old-logprob 的严格签核结论。

## 9. 控制项、通信与非 kernel 逻辑

| 项目 | 实现 | 来源 |
|---|---|---|
| TF32 / reduced-precision reduction / BLAS backend | `torch.backends.*` 设置 | TORCH；设置语义来自 vLLM BI |
| Attention `num_splits=1` | UniMatch 局部 rebind vLLM BI gate | VLLM-NATIVE + UM-WRAPPER |
| TP4 fused output AllGather | UniMatch TMA/multimem/NVLS GEMM epilogue | UM-CUSTOM + CUDA/NVLS |
| o_proj input AllGather | vLLM TP group `all_gather` | VLLM-NATIVE / NCCL |
| FSDP parameter AllGather/reduce-scatter | PyTorch FSDP distributed runtime | TORCH / NCCL |
| VeOmni EP communication | DeepEP HT | EXTERNAL |
| Weight export/layout/reload | UniRL Python布局映射 + Torch distributed + vLLM weight loader | UniRL/TORCH/VLLM |
| Stage dump | UniMatch Python hooks + `torch.save` | UM-WRAPPER + TORCH |
| Exact gate | UniRL FP32 cast、abs/max、K3、`torch.equal` | UniRL + TORCH |

精度控制源码：

- `/root/bruceszchen_gy2/MyProjects/UniMatch/unimatch/adaptor/vllm/patches/batch_invariant_precision/patch_precision.py:56-102`
- vLLM 原始 BI 设置：
  `/opt/conda/envs/unirl/lib/python3.12/site-packages/vllm/model_executor/layers/batch_invariant.py:900-989`

## 10. 容易误分类的项目

1. `batch_invariant_reductions/reductions_triton.py` 位于 UniMatch，但文件明确
   是 vLLM deterministic reductions 的自包含副本：归为 VLLM-BI-COPY。
2. `batch_invariant_norm/norm_triton.py` 也是 vLLM BI RMSNorm 副本，但当前
   `patch_norm.py` 实际 import 的是 `unimatch.ops.rmsnorm.rms_norm_fwd`，
   因此当前 RMSNorm 应归为 UM-CUSTOM。
3. Actor FA3 通过 UniMatch 暴露成 Transformers 期望的模块接口，但底层
   `flash_attn_varlen_func` 来自 `vllm.vllm_flash_attn`：归为
   VLLM-NATIVE + UM-WRAPPER。
4. TP1 QKV/o_proj/LM head 的 wrapper 位于 UniMatch，但 exact 分支直接调用
   vLLM `linear_batch_invariant`：归为 VLLM-BI-DIRECT。
5. `rotary_vllm_exact.py` 位于 UniMatch 且命名为 kernel，但当前实现是
   PyTorch BF16 slice/mul/add/cat expression，不是独立 Triton/CuTe kernel：
   归为 TORCH + UM-WRAPPER。
6. `rank_local_partial` 位于 UniMatch，但实现是显式 Python slot loop 和
   Torch FP32 tensor ops：归为 TORCH，不计入 UniMatch 自定义 device kernel。
7. `vllm_softmax_kernel` 是独立 Python/CUDA 扩展依赖，不属于 vLLM 主仓库
   的 batch-invariant 模块，也不属于 UniMatch vendored kernel。

## 11. 最终结论

当前 bitwise 对齐不是“全部使用 UniMatch kernel”，而是一个混合栈：

- Dense actor reference 大量直接复用 vLLM batch-invariant linear。
- Log-softmax/reduction 使用保存在 UniMatch 中的 vLLM BI 自包含副本。
- Attention 两侧复用 vLLM 原生 FA3，UniMatch 只固定 BI 调度合同。
- RMSNorm、router gate、TP4 dense/grouped GEMM、VeOmni local grouped
  FC1/FC2、EP-ordered combine 是 UniMatch 自研 kernel。
- embedding、top-k、排序、SwiGLU、RoPE 显式算术、routing-weight
  处理等大量逻辑来自 Torch/ATen。
- EP4 通信依赖 DeepEP，跨卡 collectives 最终依赖 NCCL/CUDA。

任何替换或开源清理都必须按上述真实来源分别处理许可证、ABI、数值合同和
逐位门禁，不能仅按源码所在仓库判断归属。
