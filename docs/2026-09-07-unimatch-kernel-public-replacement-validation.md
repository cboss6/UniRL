# UniMatch 自研 kernel 公共替代与 1024-token 验证

日期：2026-09-07

## 1. 基线与实验分支

原始严格 bitwise 基线：

- MyUniRL4：`6681ed8a8d2b4484b02d50db39ef5555cd6d1c8e`
- UniMatch：`66119fb`
- 分支：`unirl-bitwise`

公共替代实验分支：

- 两个仓库均为 `unirl-bitwise-public-replacements`
- 最终 MyUniRL4：`79d967a`
- 最终 UniMatch：`a566225`

原始 provider 实现未删除，仍可通过环境变量回退。实验分支的两个 launcher
默认选择公共 provider。

## 2. 验收标准

每个候选先做算子 micro-gate，再在累计替代栈上分别运行：

- FSDP actor world=4 / vLLM rollout TP4
- VeOmni actor EP4 / 四个 vLLM rollout TP1
- response length=1024
- `ignore_eos=true`
- old logprob 来自 actor 的整段 no-grad exact `prompt+response` forward

每个 actor rank 必须同时满足：

```text
dtype=torch.float32
finite=True
torch_equal=True
mismatch_count=0
max_absdiff_fp32=0.0
k3_mean=0.0
k3_max=0.0
```

最终累计栈额外执行：

- 每 rank 4×1024 tokens
- 一次非零 optimizer update
- EP4 full-weight export
- 四个 vLLM TP1 replica reload
- 更新后再次 4×1024 strict gate

## 3. 最终启用的公共 provider

```text
UNIMATCH_RMSNORM_PROVIDER=vllm_bi
UNIMATCH_RMSNORM_BACKWARD_PROVIDER=torch
UNIMATCH_REDUCTION_PROVIDER=vllm_bi
UNIMATCH_MOE_COMBINE_PROVIDER=torch
UNIMATCH_DENSE_PROVIDER=vllm_bi
UNIMATCH_GROUPED_PROVIDER=vllm_bi_loop
UNIMATCH_GATE_PROVIDER=vllm_bi
UNIMATCH_ROUTER_SOFTMAX_PROVIDER=vllm_bi
```

底层公开来源：

- vLLM 0.22 `batch_invariant.py`，Apache-2.0
- PyTorch 2.11 tensor/autograd/distributed APIs
- vLLM native FA3、TP group 与 MoE permute，Apache-2.0
- DeepEP HT，MIT
- NCCL/CUDA runtime

vLLM 官方 batch-invariance 文档：
<https://docs.vllm.ai/en/v0.27.0/features/batch_invariance/>

vLLM BI 源码：
<https://github.com/vllm-project/vllm/blob/3e49479c/vllm/model_executor/layers/batch_invariant.py>

## 4. 替代结果总表

| 原 UniMatch 实现 | 公共替代 | 是否直接保留旧 bits | TP4 1024 | EP4 1024 | 结论 |
|---|---|---:|---:|---:|---|
| CuTe RMSNorm forward | vLLM `rms_norm_batch_invariant` | 否 | 通过 | 通过 | 保留公共替代 |
| UniMatch 内 vLLM reduction 副本 | 直接调用 vLLM BI log-softmax/softmax/mean | 是 | 通过 | 通过 | 删除热路径复制依赖 |
| CuTe `MoeCombineEpOrdered` | Torch rank/slot 显式 FP32 左折叠 + BF16 rank 边界 | 是 | 通过 | 通过 | 保留正确性优先替代 |
| TP dense QKV/o_proj/LM head | vLLM BI local linear + NCCL AllGather | 否/共同换 bits | 通过 | 通过 | 保留正确性优先替代 |
| grouped FC1/FC2 | 每 active expert 调 vLLM BI linear + NCCL AllGather | 否/共同换 bits | 通过 | 通过 | 保留慢速 reference |
| BF16→FP32 router gate GEMM | vLLM BI BF16 linear，之后统一上转 FP32 | 否 | 通过 | 通过 | 保留公共替代 |
| `vllm_softmax_kernel.row_softmax` | vLLM BI `softmax_batch_invariant` | 否 | 通过 | 通过 | 移除独立扩展热路径依赖 |
| CuTe RMSNorm backward | Torch autograd 数学重算 | 否 | 不适用 | 更新后通过 | 保留功能替代 |

“不保留旧 bits”表示 rollout 与 actor 两侧共同切换到新的确定性黄金值，
不表示两侧不一致。

## 5. 分项结果

### 5.1 RMSNorm forward：vLLM BI

实现：

- rollout `RMSNorm.forward_cuda` 通过 provider 调用
  `vllm.model_executor.layers.batch_invariant.rms_norm_batch_invariant`
- FSDP/VeOmni actor 的 `rmsnorm_bi` forward 调用同一 API
- residual add 仍由 wrapper 在归一化前按 BF16 实体化

Micro-gate：

```text
UniMatch CuTe vs vLLM BI:
  different elements = 2 / (128×2048)
  max_abs = 0.001953125
vLLM BI row M=1 vs same row M=128:
  torch.equal=True
```

说明：该替代改变历史黄金 bits，但 vLLM primitive 自身跨 batch 严格不变。

提交：

- UniMatch `bae9955`
- MyUniRL4 `ac5da78`

1024 日志：

- TP4：`logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_200645.log`
- EP4：`logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_201342.log`

### 5.2 Reduction/log-softmax：直接调用 vLLM

原 `reductions_triton.py` 是 vLLM BI reduction 的自包含复制。新
`unimatch.functional.reductions` 直接分派到 vLLM 安装包，不再让签核热路径
依赖该复制实现。

Micro-gate：

```text
shape = [17, 151936], FP32
copied vs direct:
  torch.equal=True
  different=0
  max_abs=0
direct M=1 vs M=17 first row:
  torch.equal=True
```

提交：

- UniMatch `9c78046`
- MyUniRL4 `a5485ed`

1024 日志：

- TP4：`logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_202117.log`
- EP4：`logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_202742.log`

### 5.3 EP-ordered combine：纯 Torch

公开替代没有复制 CuTe kernel。它显式执行：

```text
for rank in ascending EP rank:
    rank_acc = FP32 zeros
    for slot in ascending top-k slot:
        rank_acc = rank_acc + FP32(valid contribution)
    total = total + FP32(BF16(rank_acc))
output = BF16(total)
```

Micro-gate：

```text
M=33, H=2048, topk=8, experts=128
EP1: Torch vs CuTe torch.equal=True
EP4: Torch vs CuTe torch.equal=True
```

提交：

- UniMatch `5188f93`
- MyUniRL4 `78be7a5`

1024 日志：

- TP4：`logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_203628.log`
- EP4：`logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_204557.log`

该实现约比 fused combine 慢，但 top-k=8、EP≤4 时可作为透明、可审计的
正确性 reference。

### 5.4 TP dense：vLLM BI linear + NCCL

替代 topology：

- QKV：每 rank 对已有本地 weight rows 调 `linear_batch_invariant`，不通信
- o_proj：先用 vLLM TP group AllGather 完整 K input；每 rank 计算完整 K
  的本地输出列；再 AllGather 输出列
- LM head：每 rank 计算本地 vocab rows；再 AllGather 完成词表

所有通信只搬运已完成列，不做数值归约。

提交：

- UniMatch `51cd52d`
- MyUniRL4 `18c2416`

1024 日志：

- TP4：`logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_205713.log`
- EP4：`logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_210650.log`

EP4 rollout 为 TP1，原路径本就已调用 vLLM BI linear；该 gate 用于确认累计
provider 没有回归。

### 5.5 Grouped experts：vLLM BI per-expert loop

替代 topology：

1. rollout 继续使用 vLLM native `moe_permute`
2. 只遍历当前有 token 的 experts
3. 每个 expert 的 FC1/FC2 调 vLLM `linear_batch_invariant`
4. TP4 对每个 expert 的已完成输出列执行 NCCL AllGather
5. SwiGLU、routing multiply 使用 Torch
6. 最终使用 Torch EP combine

VeOmni actor 在 DeepEP dispatch 后对 32 个 rank-local experts 使用同一
per-expert vLLM BI loop。

提交：

- UniMatch `8300360`
- MyUniRL4 `d77233b`

验证：

- TP4 response=1 smoke：通过
- TP4 response=1024：
  `logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_212717.log`
- EP4 response=1024：
  `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_214144.log`

该方案数值正确但 kernel launch 与 collective 数量大，只适合 correctness
reference、回归门禁或小 batch，不建议直接替代生产 fused grouped kernel。

### 5.6 Router gate：vLLM BI BF16 linear

公开 vLLM API没有“BF16 input、未舍入 FP32 accumulator output”。替代方案
改变共同数值合同：

```text
BF16 input/weight
→ vLLM linear_batch_invariant
→ BF16 logits
→ FP32 upcast
→ softmax/top-k
```

Micro-gate：

```text
vLLM BI M=1 vs M=128 first row: torch.equal=True
custom FP32-tail vs public BF16:
  max_abs=0.481719970703125
  random 128-row top-k ID mismatch rows=27
```

因此它明显改变模型 routing 与生成轨迹，但 rollout 和 actor 共同使用后仍保持
严格一致。

提交：

- UniMatch `8002038`
- MyUniRL4 `250cc38`

1024 日志：

- TP4：`logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_215204.log`
- EP4：`logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_220516.log`

如果要求保留旧 FP32-tail 模型语义，则这项不能替代；需要继续保留原 kernel，
或向 vLLM upstream 增加一个公开 FP32-output BI linear API。

### 5.7 Router softmax：vLLM BI

Micro-gate：

```text
shape=[4096,128], FP32
vllm_softmax_kernel vs vLLM BI:
  different=133908
  max_abs=7.450580596923828e-09
  top-k ID mismatch rows=0
vLLM BI M=1 vs M=4096 first row:
  torch.equal=True
```

提交：

- UniMatch `a569e48`
- MyUniRL4 `b07d37b`

1024 日志：

- TP4：`logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_221420.log`
- EP4：`logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_222655.log`

### 5.8 RMSNorm backward：Torch autograd

Torch provider按数学定义重建 RMSNorm：

```text
variance = mean(float(x)^2)
normalized = float(x) * rsqrt(variance + eps)
output = BF16(normalized * float(weight))
torch.autograd.grad(output, [x, weight], grad_output)
```

Micro-gate：

```text
forward equal=True
Torch grad_x finite=True
Torch vs CuTe grad_x max_abs=0.03125
Torch grad_w finite=True
Torch vs CuTe grad_w max_abs=0
```

它不是旧 CuTe backward 的 bitwise replacement，但给出公开、可维护、数学一致
的训练 backward。

提交：

- UniMatch `a566225`
- MyUniRL4 `79d967a`

最终非零更新日志：

- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_231126.log`

结果：

```text
rollout 1:
  reward=0.3125
  grad_norm=1.6778
  old|max_absdiff=0
  old K3 mean/max=0

updated replica digest:
  e0ffc2be7ee0d6f655bc7af624487d12236fe8dad656028a14a8f4153e73d5d7

rollout 2 after reload:
  reward=0.5000
  grad_norm=3.0135
  old|max_absdiff=0
  old K3 mean/max=0
```

## 6. 最终累计公共栈

最终 no-grad bitwise forward 不再需要下列 UniMatch 自研 device kernel：

- CuTe RMSNorm forward
- UniMatch reduction copy
- TP column-parallel dense GEMM/GEMV
- TP column-parallel grouped GEMM/GEMV
- BF16→FP32-out router gate GEMM
- CuTe EP-ordered combine
- `vllm_softmax_kernel`

仍保留的 UniMatch 代码主要是：

- provider 选择和 vLLM/Transformers/VeOmni 接线
- Qwen3 权重布局与 TP/EP topology orchestration
- DeepEP autograd/lifecycle wrapper
- stage dump 与 fail-closed contract
- 原高性能 kernel provider，作为可选 fallback

底层计算来自公开 vLLM、Torch、DeepEP 和 NCCL/CUDA。

## 7. 性能与实现难度

| 项目 | 公共实现难度 | 当前公共替代性能 | 若要求生产性能 |
|---|---:|---:|---|
| RMSNorm forward | 低 | 良好 | 直接保留 vLLM BI |
| Reductions/log-softmax | 低 | 良好 | 直接保留 vLLM BI |
| EP combine | 低 | 中等偏慢 | 可向 vLLM/Triton upstream 提交固定 rank/slot fold |
| Dense QKV/o_proj/LM | 中 | 较慢，额外 NCCL | 等待/推动 vLLM 公共 column-parallel BI + AllGather |
| Grouped experts | 中 | 很慢 | 需要 vLLM upstream 的 batch-invariant grouped GEMM |
| Router gate FP32 tail | 高 | vLLM BF16 fallback 性能良好但改变路由 | upstream 增加 FP32-output BI linear |
| Router softmax | 低 | 可接受 | 保留 vLLM BI |
| RMSNorm backward | 低 | 慢、内存较高 | 可保留 Torch reference，性能路径需独立优化 |

最难的不是重新写 Python 表达式，而是同时满足：

- 固定 K accumulation
- FP32 tail 或指定 BF16 materialization
- TP/EP 列布局
- grouped expert 的动态 M
- 通信 epilogue
- CUDA graph/lifecycle

如果不要求保持旧黄金 bits，本次实验证明可以通过“两侧共同切换公共 primitive”
完全移除签核前向对 UniMatch 自研 device kernel 的硬依赖。若同时要求生产性能
和旧 bits，则 dense/grouped/gate 仍需要 upstream 新 API 或保留专用 kernel。

## 8. 结论

### 正确性优先

已实现。最终公共栈在 TP4、EP4 1024 tokens 以及非零更新后 reload gate 中
严格通过。

### 性能优先

尚不能宣称公共栈优于原 UniMatch：

- per-expert loop 与非融合 NCCL 明显更慢
- Torch combine/backward增加 kernel launch 和临时 tensor
- vLLM BI gate 改变路由黄金值

建议保留两个 profile：

```text
public/reference:
  全部公共 provider，用于审计和 correctness

unimatch/performance:
  原 fused dense/grouped/gate/combine provider，用于生产性能
```

后续若公共 vLLM 增加 batch-invariant grouped GEMM、FP32-output linear 与
column-parallel AllGather API，可逐项替换 performance profile，而不必再复制
或独立维护 kernel。
