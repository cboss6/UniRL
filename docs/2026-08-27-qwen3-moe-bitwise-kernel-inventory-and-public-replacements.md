# Qwen3-MoE UniRL 逐位一致内核清单与公共替代方案评估

日期：2026-08-27

严格 old-logprob 语义修订：2026-09-07

按实际实现来源分类的简明清单见
[`2026-09-07-qwen3-moe-bitwise-op-kernel-source-inventory.md`](2026-09-07-qwen3-moe-bitwise-op-kernel-source-inventory.md)。

公共 vLLM/Torch provider 的逐项替换与 1024-token 验证见
[`2026-09-07-unimatch-kernel-public-replacement-validation.md`](2026-09-07-unimatch-kernel-public-replacement-validation.md)。

## 1. 范围与已签核结果

本文档的历史基线为：

- UniMatch 分支 `unirl-bitwise`，提交 `deeb509`
- UniRL 分支 `unirl-bitwise`，提交 `28f16ec`

2026-09-07 的整段 old-policy forward 修订当前位于上述工作区的未提交
改动中；仅检出这两个历史提交不能获得本节新增的严格门禁。

已签核配置如下：

- 模型：Qwen3-30B-A3B，BF16
- rollout：vLLM 0.22，TP=4，EP=1，未量化 Triton MoE 后端
- actor：Transformers 5.6，FSDP 全局规模 4
- 长序列门禁中的提示词长度：16
- 长序列门禁中的响应长度：1024
- 分块预填充：禁用
- 前缀缓存：禁用
- vLLM 即时执行：启用
- actor old-logprob 比较路径：no-grad exact，对
  `prompt+response` 执行整段 packed/padded teacher-forcing forward

相关模型结构参数：

- 隐藏层大小 2048
- 32 个查询头、4 个 KV 头、头维度 128
- TP4 QKV 每个 rank 的输出分片为 1280 列
- 路由器 gate `[128, 2048]`，输入/权重为 BF16，logits 为 FP32
- 128 个专家，top-k 8，中间层大小 768
- LM head `[151936, 2048]`，每个 TP rank 37984 个词表行

2026-09-07 的 1024-token 整段 old-policy 门禁在全部四个 actor rank 上均报告
`torch_equal=True`、`mismatch_count=0`、FP32 `max_absdiff=0`，且
K3 mean/max 均为零。此前与 KV 长度 128–130 对应的调用 112–114
及 1,440 次逐层张量比较属于历史 decode-topology 诊断证据，不再作为
old-logprob 的签核路径。

当前严格结论为：

> vLLM rollout 对数概率与 actor 对同一 `prompt+response`
> 执行整段 no-grad exact forward 所得 FP32 old log-probabilities
> 逐元素相等。

这尚不意味着每个承载梯度的训练前向传播与
反向传播 kernel 都和 vLLM 逐位一致。UniRL 将整段 no-grad exact
old-policy forward 与承载梯度的 GRPO replay 分开记录。

## 2. 已启用的补丁栈

启动器启用了以下 vLLM 补丁：

- `qwen3_moe`
- `batch_invariant_reductions`
- `batch_invariant_norm`
- `batch_invariant_precision`
- `batch_invariant_attention`

`VLLM_BATCH_INVARIANT=0` 仍为全局设置。对于 vLLM 0.22，注意力补丁
还安装了一个后端局部环境代理，使 FA 调度器直接读取
`envs.VLLM_BATCH_INVARIANT` 时解析为 true，而不会启用无关的
全局 vLLM batch-invariant 路径。

它启用了以下 Qwen3-MoE 组件：

- `experts_grouped`
- `qkv_local`
- `oproj_col`
- `lm_head_col`
- `gate_fp32`
- `router_hf`

FSDP 模型在构建 Transformers 模型之前导入
`unimatch.adaptor.fsdp.bootstrap`。它启用了：

- Qwen3-MoE 模块补丁
- ATen linear 与 softmax 的精确 CUDA 实现
- 将 vLLM FA3 用作 Transformers 注意力后端
- 路由专家的 TP=4 算术模拟

`UNIMATCH_MEGATRON=0`、`UNIMATCH_TRAIN_EP_SIZE=1` 与
`UNIMATCH_MOE_SUM=bf16` 均属于已签核约定。

### 2.1 探测路径与梯度路径

该实现有三种不同的执行模式：

- no-grad old-policy 对齐探测：整段 packed/padded teacher-forcing、
  精确 ATen linear 模式、FA3、确定性 LM head，以及确定性 log-softmax
- alignment-gate-only：使用缓存的探测所得 log-probabilities，且不执行
  反向传播
- 梯度 GRPO replay：打包或填充的
  teacher-forcing，禁用 exact dense-linear
  模式，使用自定义 MoE 重新计算反向传播、RMSNorm 自定义反向传播，以及
  PyTorch math-SDPA 注意力反向传播

Qwen3 RMSNorm、RoPE、路由器模块、MoE 严格前向传播与 FA3 前向传播
补丁均安装在模块层级。精确 dense ATen linear 行为被限制在 no-grad
old-policy forward 中。旧的单 token 分页 replay 仍可用于历史诊断，
但不参与 old-logprob 门禁。进程全局 ATen softmax 注册不会查询
`exact_mode`；启用后，受支持的 router softmax rows 在 probe 与训练前向传播中
都会使用已注册的 row-softmax 实现。

FSDP 与 vLLM 必须处于不同进程，因为两侧都会安装
进程全局 ATen CUDA 实现。FSDP 兼容性检查当前
仅接受 Transformers 5.6.x 与 5.16.x。

## 3. 按算子逐项说明实现

### 3.1 嵌入查找

当前 vLLM 侧：

- 使用原生 `VocabParallelEmbedding`。
- 每个 TP rank 在自己的词表分片中查找、屏蔽非本地 tokens，并使用
  原生 TP 归约。
- UniMatch 不替换此 kernel。

当前 FSDP 侧：

- 使用原生 Transformers embedding，逻辑
  FSDP 模型上保留完整 embedding table。

一致原因：

- 对于每个 token，恰好只有一个 TP rank 提供非零 embedding row。
  因此 TP sum 不会引入浮点归约顺序
  歧义。

公共替代方案：

- 可直接复用。不需要私有 kernel，也不需要新的 Triton/CuTe
  实现。

### 3.2 RMSNorm 与融合 add-RMSNorm

当前 vLLM 侧：

- `batch_invariant_norm` 替换 `RMSNorm.forward_cuda`。
- 它调用 `unimatch.ops.rmsnorm.rms_norm_fwd`，
  后者由 UniMatch CuTe DSL
  RMSNorm kernel 支撑。
- 残差模式计算 `added = x + residual`，返回固定顺序的 norm
  以及 BF16 残差和。
- 该 kernel 每行使用一个 CTA，其归约树取决于隐藏层大小，
  而不取决于批大小。

当前 FSDP 侧：

- `Qwen3MoeRMSNorm.forward` 调用
  `unimatch.functional.rmsnorm.rmsnorm_bi`。
- 前向传播使用相同的 CuTe DSL 归约。
- 训练反向传播使用配套的自定义 RMSNorm 反向传播实现。

一致原因：

- 两侧使用相同的 FP32 平方和顺序和相同的 BF16 输出
  实体化。
- vLLM 避开 Oink 及其他归约调度会随
  运行时形状改变的融合路径。

公共替代方案：

- 为保留已经签核的历史 tensor bits，必须保留当前固定的
  归约树，或在 Triton/CuTe 中独立复现；
  不能假定 UniMatch 的 CuTe tree 与 vLLM 的 Triton tree 逐位一致。
- 如果允许更改黄金 bits，两侧可一同切换至
  vLLM 的 batch-invariant Triton RMSNorm 并建立新的 K3=0 签名，
  前提是核验源代码许可证与导出 API 的稳定性。
- 只有在归约树被固定且完整门禁阶梯重新签核后，
  小型公共 Triton 逐行 RMSNorm 才足够。
- 原生 `torch.nn.functional.rms_norm` 不是已签核的直接替代：
  后端选择与归约树并无契约性固定。
- 残差加法必须在归一化前保持 BF16 实体化；
  原生 fused add-RMSNorm 不会自动等价。

调试边界：

- `layer_input`
- `attention_input`
- `moe_input`

### 3.3 Q、K 与 V 投影

当前 vLLM 侧：

- `qkv_local` 替换 `QKVParallelLinear.forward`。
- 它以仅本地、`variant=base`、
  `auto_tile=True` 模式使用 `TpColumnParallelDense`。
- 每个 rank 仅计算其本地 Q/K/V 头分片。
- GEMM 之后没有 AllGather 或跨 rank 算术。
- 自定义 GEMM 针对 prefill 与 decode 的
  M 值固定完整 K 累加顺序。

当前 FSDP 精确 replay 侧：

- 进程全局 ATen linear override 仅在 eval 或
  no-grad 模型前向传播时进入精确模式。
- 大多数 dense 投影调用 vLLM 的 `linear_batch_invariant`。
- FSDP 模型计算完整 Q/K/V 张量。比较工具在检查相等性前
  拼接四个 vLLM TP 分片。

承载梯度的 FSDP 侧：

- 普通训练前向传播会禁用精确 linear 模式。
- 回退方案为带 PyTorch 反向传播的 PyTorch matmul/linear。

已签核 replay 的一致原因：

- 输出列相互独立。
- 两种实现使用相同的 batch-invariant 完整 K 累加约定。

公共替代方案：

- 正确性优先：TP 分片与完整 FSDP 投影均调用 vLLM 的公共
  批次不变 linear kernel。
- 原生 PyTorch matmul 不是已签核替代，因为 cuBLAS 可能针对不同
  M 值选择不同算法。
- 如果不能接受依赖 vLLM helper，则需要固定调度 Triton
  GEMM。QKV 不需要自定义 NVLS/AllGather kernel，
  因为 QKV 仅执行本地计算。

调试边界：

- `q_proj`
- `k_proj`
- `v_proj`
- `q_norm`
- `k_norm`

### 3.4 Q/K RMSNorm

当前实现：

- vLLM 与 FSDP 都将 Q/K norm 路由至上述同一
  批次不变 RMSNorm 约定。
- 为进行比较，TP 头分片沿头维度拼接。

公共替代方案：

- 结论与 decoder RMSNorm 相同：vLLM 的确定性 Triton kernel 是
  合适的公共依赖；否则需独立实现 fixed
  逐行 Triton 归约。

### 3.5 RoPE

当前 vLLM 侧：

- UniMatch 在模型构建前安装规范 RoPE 约定。
- 根据模型常量重建 FP32 逆频率。
- 该实现索引 BF16 cos/sin 缓存，并用显式即时乘法、减法、加法
  与 concatenate 操作应用 Neox rotation。

当前 FSDP 侧：

- `Qwen3MoeRotaryEmbedding.forward` 在
  GPU 上重建 FP32 逆频率，在 BF16 autocast
  外计算 phase、cos 和 sin，随后将 cache
  转换为激活 dtype。
- Transformers 应用 eager rotation。

一致原因：

- 两侧使用相同的逆频率构造、缓存 dtype、
  位置索引、Neox 布局与即时算术顺序。
- stage dump 发现的第一个 prefill 差异正是缺少 vLLM 规范约定。

公共替代方案：

- 直接 PyTorch/ATen 操作即已足够，也是推荐的
  开源回退方案。
- 原生 vLLM CUDA RoPE 与已签核 eager contract 不可互换，
  除非版本特定的 micro-gate 证明二者相等。
- 不需要新的 Triton 或 CuTe kernel。

调试边界：

- `q_rope`
- `k_rope`

### 3.6 注意力核心

当前 vLLM 侧：

- 使用 vLLM 内置的 FA3 kernel。
- 解码使用 16-token 分页 KV 缓存、块表与 `seqused_k`。
- 批次不变注意力门禁与 vLLM-0.22 后端局部环境代理
  约束调度，包括 split-KV 行为。

当前 FSDP 侧：

- UniMatch 通过 Transformers 期望的
  module name 暴露 vLLM 内置的 FA3。
- 当前 old-policy 门禁对 `prompt+response` 执行完整的 packed/padded
  teacher-forcing forward；其 attention 调用使用 FA3 varlen，
  并显式提供 Q 与 KV cumulative lengths、
  deterministic mode、`num_splits=1` 以及 FA3。
- 历史诊断用 no-grad 单 token 解码可将 Transformers 缓存打包为 16-token 页，
  并使用 identity block table 与
  `seqused_k` 调用 FA3，以匹配 vLLM 的 paged
  布局。
- 该 paged 路径被限制为不请求梯度的 one-token 调用，现不参与
  rollout-vs-old-logprob 签核。
- 自定义 autograd 回退方案通过 PyTorch
  math SDPA 重新计算 attention 反向传播；反向传播不属于 rollout K3。

一致原因：

- FA3 contiguous 与 paged KV 在 KV length 128 之前恰好一致，但在
  129 时出现差异。仅匹配 `num_splits` 并不足够。
- 上述差异说明历史逐层 decode tensor 对比需要匹配 paged memory topology。
  但 old-logprob 的正确验收对象是实际整段 actor forward；2026-09-07
  的 TP4 与 EP4 1024-token selected-logprob 门禁已证明该路径最终 FP32
  logprob 与 rollout 相等，不据此宣称所有 attention 中间 tensor 相等。

公共替代方案：

- 两侧复用 vLLM 内置的 FA3 primitive。这是首选
  方案；需核验其分发许可证和受支持的公共 API。
- PyTorch SDPA、原生 Transformers FA3 连续模式，以及普通即时
  softmax 注意力均不能逐位替代 vLLM paged decode。
- 如果无法重新分发或依赖 vLLM FA3，则必须实现一个公共的 deterministic
  paged attention kernel 并在两侧使用。这是真正的
  自定义 Triton/CuTe 需求，无法通过改变 Python expression order 修复。

调试边界：

- `v_attn`
- `attn_core_output`
- `oproj_input`

### 3.7 注意力输出投影

当前 vLLM 侧：

- 原生 o_proj 为行并行，会先执行 GEMM partials，随后执行
  AllReduce。
- UniMatch 首先在 K-sharded attention
  input 上使用 vLLM 的 TP-group AllGather，随后运行带融合 AllGather 的
  列并行 dense GEMM。
- 经过验证的实现使用
  `TpColumnParallelDenseAuto`；
  prefill 路由至 WGMMA GEMM，小 decode M
  可能路由至 `mma.sync` GEMV。
- Multimem/NVLS 提供跨 rank 回写与设备侧屏障。

当前 FSDP 精确 replay 侧：

- 完整 o_proj weight 由精确 ATen linear 路径计算。

一致原因：

- 两侧均以相同固定顺序计算完整 K。
- AllGather 只移动已经完成的输出列；不会对部分和进行数值 reduction。

公共替代方案：

- 正确性优先：使用 NCCL AllGather 收集 input 与所需权重分片，在完整 K 上调用
  相同的公共 batch-invariant linear kernel，随后 AllGather
  输出列。该方案较慢，但可避免私有 NVLS/CuTe 代码。
- 原生行并行 GEMM 加 NCCL AllReduce 不等价，因为它
  引入了不同的 FP32/BF16 部分和顺序。
- 若要在不依赖 UniMatch 的情况下获得生产性能，则需要独立实现
  列并行-GEMM-plus-AllGather Triton/CuTe kernel。
  multimem epilogue 是性能要求，而非正确性
  要求。

调试边界：

- `oproj_input`
- `oproj_output`
- `attention_output`

### 3.8 路由器 gate GEMM

当前 vLLM 侧：

- 复制的 BF16 gate 权重由
  `DenseGemmBF16Fp32Out` 计算。
- 该 kernel 固定 K accumulation order，并将未舍入的 FP32
  累加器暴露给 routing。

当前 FSDP 精确 replay 侧：

- 特殊的 `(N=128, K=2048)` ATen linear case 调用
  `dense_bf16_fp32out`。
- 因此 Router softmax 接收到相同的 FP32 logits。

承载梯度的 FSDP 侧：

- exact mode 未启用时，gate 的 linear 调用回退至 PyTorch matmul。

一致原因：

- 返回 FP32 tail 至关重要。在 top-k 前将 logits 舍入至 BF16
  可能改变专家标识，而不只是引入很小的输出误差。

公共替代方案：

- 原生 vLLM 或 PyTorch linear API 均无法同时保证已签核的
  固定 K order 与未舍入 FP32 输出。
- 正确性原型可以使用显式串行化的 FP32 FMA 链，但
  必须针对已签核 kernel 进行 micro-gate，且速度会慢到无法接受。
- 实用实现需要独立实现一个小型
  BF16 输入/FP32 输出固定调度 Triton 或 CuTe GEMM。这是无法通过
  无条件原生调用替换的 kernel 之一。

调试边界：

- `router_logits`

### 3.9 路由器 softmax、top-k 与重新归一化

当前 vLLM 侧：

- `router_hf` 按以下顺序执行：
  1. 来自 `vllm_softmax_kernel.row_softmax` 的 FP32 row softmax
  2. `torch.topk`
  3. 显式计算 selected-weight sum 并执行 division
- 运算顺序与 Transformers actor 一致。
- `vllm_softmax_kernel` 是外部运行时依赖，未由 UniMatch vendored。

当前 FSDP 侧：

- router 补丁执行相同的完整 softmax、top-k 与重新归一化
  顺序。
- ATen softmax override 将受支持的行大小分派给相同的
  `vllm_softmax_kernel.row_softmax`。
- 与 exact linear 不同，此进程全局 softmax override 在
  承载梯度的训练前向传播期间也会启用。

公共替代方案：

- 两侧复用一个公共 row-softmax kernel。如果
  `vllm_softmax_kernel` 无法随项目发布，可使用一个小型、直接的
  固定行 Triton kernel。
- 只要两侧接收逐位相等的 FP32 logits，并使用相同形状/顺序，
  就可以直接复用 `torch.topk` 与显式重新归一化。
- 原生 vLLM 融合 `topk_softmax` 不能直接替代这一已签核的
  HF 运算顺序。

### 3.10 专家分派与填充

当前 vLLM 侧：

- 使用 vLLM `moe_permute` 创建专家连续 rows。
- 使用设备侧计划，将每个专家分段 padding 至 128 rows 的倍数。

当前 FSDP 侧：

- 展开 token slots，并使用稳定专家排序与 `bincount`。
- 按 rank 顺序模拟四个 TP rank shards。

一致原因：

- 两条路径保留相同的 token/slot-to-专家映射。填充 rows 为
  zero，并在 combine 前移除。

公共替代方案：

- PyTorch 稳定排序、gather/scatter 与 bincount 可提供正确性优先的
  实现。
- 经过 row-map 约定门禁验证后，可复用 vLLM 的公共 permute kernel
  以提高性能。
- 分派从根本上不需要私有自定义 kernel。

### 3.11 分组专家 gate/up 与 down GEMM

当前 vLLM 侧：

- merged gate/up 与 down 均使用 `TpColumnParallelGroupedAuto`。
- Prefill 使用 grouped WGMMA GEMM；
  较小的 decode workloads 可能使用 grouped
  `mma.sync` GEMV 双生实现。
- Gate/up 通过融合 AllGather 写出完整拼接的 `[gate | up]` 结果。
- Down 使用列分片输出权重，并同样生成完整收集
  结果。
- 经过验证的 multimem 路径使用 NVLS symmetric buffers 与 graph-safe
  设备侧栅栏。
- block 层级 `moe_grouped_expert_forward`
  override 是当前热路径；
  另外安装的 `GroupedCPExperts` 对象主要用于保留
  生命周期与权重重分片集成。

当前 FSDP 侧：

- 严格前向传播按 expert 稳定地对 rows 分组。
- 对每个模拟 TP rank 及每个 expert，使用对应的
  权重分片运行 `F.linear`。
- rank 分片输出按 TP rank 顺序拼接。
- 自定义自动求导 wrapper 在反向传播
  中重新计算可微 PyTorch 路由图。
- 严格 grouped 前向传播在 probe 与训练模式中都会使用。其内部
  `F.linear` 调用仅在 no-grad 精确模式中使用精确 ATen linear；
  承载梯度的路径回退至 PyTorch matmul，并从
  重新计算的路由图获得梯度。

一致原因：

- 自定义 vLLM kernel 复现 FSDP rank-packed 列并行
  accumulation 与 BF16 实体化，而不是原生 vLLM 融合专家
  算术。

公共替代方案：

- 正确性优先：在 vLLM 侧 gather 或流式传输所需 TP 权重分片，
  并运行与 FSDP 相同的稳定按专家/按模拟 rank 循环，且两侧使用
  相同的公共批次不变 linear provider。普通
  `F.linear`/cuBLAS 无法应对变化的按专家 M 值。
  该方案避免了私有分组 kernel，但预计会慢很多，而且
  可能需要层本地 scratch 或流式 gather 才能满足内存限制。
- 如果没有 gate 证明其 GEMM 累加
  与合并顺序符合 FSDP 约定，
  则复用原生 vLLM FusedMoE 并不足够。
- 生产级 TP4 实现需要独立开发
  分组列并行 GEMM/GEMV，控制累加并提供
  通信 epilogue。Triton 可提供可移植初版；CuTe DSL
  适合 WGMMA 与 NVLS 性能优化。
- 正确性并不需要 NVLS multimem；NCCL AllGather 是有效的慢速
  通信替代方案，因为收集过程不执行数值归约。

### 3.12 SwiGLU

两侧当前实现：

- 拆分已收集的 BF16 gate/up 列。
- 以 BF16 行为实体化 `F.silu(gate)`，并与 BF16 `up` 相乘。
- `UNIMATCH_MEGATRON=0`；FP32 Megatron activation 路径处于未启用状态。
- 虽然较早的注释提到融合 vLLM activation，但启用的
  vLLM 实现同样执行此 eager PyTorch expression。

公共替代方案：

- 直接使用 PyTorch `F.silu(gate) * up` 即已足够，且当前已经在使用。
- 只有在 micro-gate 确认相同中间舍入后，才能复用
  vLLM 的融合 activation。
- 正确性不需要新的自定义 kernel。

调试边界：

- activation 包含在 grouped gate/up
  与 down 之间；需要时可在阶段转储中
  增加 `activation`。

### 3.13 路由权重乘法与专家合并

当前实现：

- 专家输出乘以路由权重，并实体化为 BF16。
- `MoeCombineEpOrdered` 执行确定性 slot 累加。
- 在 `UNIMATCH_TRAIN_EP_SIZE=1` 下，它是一条 FP32 slot 顺序累加链，
  随后转换为 BF16。
- EP>1 支持还会在跨 rank FP32 累加前，
  对每个 EP rank 建模一个 BF16 舍入边界。

公共替代方案：

- 对于已签核的 EP=1 配置，按固定
  top-k slot 顺序执行显式 Python/PyTorch
  loop，即可复现 FP32 链与末端 BF16 转换。必须
  进行 micro-gate，因为 `torch.sum(dim=...)` 可能选择不同的归约树。
- 对于 EP=1，vLLM 原生 `moe_sum` 也是候选方案，前提是贡献项已恢复为
  token-major slot 顺序，且算子门禁确认相同的
  左折叠/转换约定。
- 对于 EP>1，按升序显式循环 rank groups，并插入 BF16
  rank 边界转换。这仍是正确性优先的 PyTorch 实现。
- 建议使用紧凑的 Triton combine kernel 提升性能，但对于当前 EP=1
  约定，从根本上不需要新的私有 CuTe kernel。
- 原生 vLLM `moe_unpermute` 或平坦的 `torch.sum` 不是已签核替代。

调试边界：

- `moe_output`

### 3.14 最终 norm 与 LM head

最终 norm：

- 使用上述相同的 RMSNorm contract。

当前 vLLM LM head：

- 使用带融合 AllGather 的列并行 dense kernel。
- Gather 后词表列仍保持规范顺序。

当前 FSDP 精确 replay LM head：

- 使用完整权重精确 ATen linear 路径。

公共替代方案：

- 正确性优先：对 vLLM 词表分片与 FSDP 完整矩阵使用相同的
  公共 batch-invariant linear kernel，随后对已完成的
  词表列执行 NCCL AllGather。
- 原生 cuBLAS linear 对 prefill/decode M 变化没有签核保证。
- 仅性能需要新的融合通信 GEMM，
  正确性并不需要。

调试边界：

- 当 `UNIMATCH_LM_DEBUG=1` 时，LM debug
  会打印所选 logits 与最高 logits。

### 3.15 Log-softmax 与所选 token 对数概率

当前 vLLM 侧：

- `batch_invariant_reductions` 为 softmax、log-softmax 与 mean
  安装 deterministic Triton reductions。
- vLLM 0.22 `compute_token_logprobs` 被明确替换为 FP32
  `log_softmax(...).gather(...)`。
- 所选 token log-softmax 是该配置中已经确认的热路径用法；
  router softmax 则直接调用外部 row-softmax kernel。

当前 FSDP 精确 replay 侧：

- UniRL 在 FP32 logits 上调用相同的 Triton log-softmax 实现，并
  收集所选 token。

公共替代方案：

- 首选：在核验源代码许可证/API 稳定性后，两侧复用 vLLM 的公共
  batch-invariant log-softmax 实现。
- 否则，实现小型固定行 Triton 归约。
- `torch.log_softmax` 只有通过跨 M 与
  跨框架逐位门禁后才能作为慢速回退候选；它并非当前已签核实现。

### 3.16 TP 集合通信与对称缓冲区

当前实现：

- QKV 不含集合通信。
- o_proj、grouped experts 与 LM head 使用列并行已完成列，
  随后采用 AllGather 语义。
- UniMatch 将跨 rank 回写融合进 TMA 或 multimem/NVLS GEMM epilogue，
  并使用设备侧栅栏。

公共替代方案：

- NCCL AllGather 是保持正确性的替代，因为它移动
  已经舍入的值，并不对其求和。
- NCCL AllReduce 与已签核列并行
  topology 不可互换。
- Symmetric memory、NVLS 与自定义 graph-safe 屏障属于性能
  特性。慢速 eager 公共实现不需要它们。

### 3.17 权重布局与重新加载

当前实现：

- FSDP 将完整具名权重导出至 dtype buckets。
- UniRL 收集 rank 载荷s，并发送至全部 vLLM TP workers。
- 自定义加载器保留 QKV 与 gate/up 分片顺序。
- 行并行 w2/o_proj 权重会重塑为精确 kernels
  所期望的列并行 layout。
- Reload 在返回前同步 CUDA 工作。

公共替代方案：

- 这主要是 Python 布局与通信逻辑，而非私有
  数值 kernel。
- 在满足常规框架 API 兼容性的前提下，可直接开源。

### 3.18 全局精度与调度控制

当前实现：

- 禁用 TF32。
- 禁用低精度 BF16/FP16 归约。
- 固定 cuBLAS workspace 行为。
- 在 vLLM backend 中选择性约束 attention split scheduling。
- 任何可选图捕获前都会预热自动分块 CuTe kernels。

公共替代方案：

- 精度 flags 与 cuBLAS workspace
  configuration 都是公共 PyTorch/CUDA
  设置。
- 即时正确性回退方案不需要 NVLS 缓冲区、graph-safe 屏障、
  持久输出地址或自动分块预热。
- 只有重新引入 CUDA graphs 或 fused
  communication 后，这些性能机制才会变得相关，且必须单独接受 gate 验证。

## 4. 替代优先级

本节采用以下概念性替代标签：

- A：直接的公共/原生替代方案能够保留当前算术
  约定
- B：存在正确性优先的公共组合方案，但速度较慢
- C：公共 primitive 只有在固定 schedule/layout 并添加 wrapper 后才可使用
- D：要保留当前已签核的历史 bits，必须保留或
  独立复现 kernel 的固定归约树

跨侧相等与历史 bit 保留是不同目标。为保留历史结果而分类为 D 的
kernel，有时可以在两侧一同替换，并重新签核为新的 K3=0 黄金结果。

### 4.1 可直接复用的原生/公共原语

以下各项不需要新的私有 kernel：

- 嵌入
- 显式即时 RoPE
- PyTorch top-k 与 router 重新归一化
- 稳定-sort 分派与 gather/scatter
- 即时 SwiGLU
- vLLM 内置 FA3，包括单 token replay 的分页 KV
- NCCL AllGather
- 权重布局与重新加载逻辑

### 4.2 两侧复用同一个 vLLM 确定性原语

如果可以接受依赖 vLLM，则不应重复实现以下各项：

- 批次不变 RMSNorm
- 用于 QKV/o_proj/LM head 正确性路径的批次不变 dense linear
- 逐行 softmax
- log-softmax

分发前需核验每个依赖项的许可证与导出 API。

### 4.3 正确性优先的 PyTorch 回退方案

以下方案可行，但速度明显更慢，且需要专用逐位门禁：

- 按专家、按模拟 TP rank 执行的 `F.linear` 循环
- 围绕完整 K 列并行 dense operations 的 NCCL AllGather
- 用于 EP=1 的显式 top-k slot 累加循环
- 用于 EP>1、带 BF16 边界的显式 EP-rank 括号循环

### 4.4 实用性能所需的独立 Triton/CuTe 实现

如果无法分发 UniMatch kernels，以下是优先
重实现项：

1. BF16 输入/FP32 输出路由器 gate GEMM。除非接受极慢的串行化
   原生 reference，否则正确性需要此项。
2. Grouped 列并行 gate/up 与 down GEMM/GEMV。原生 loop 可以
   正确，但无法作为实用的 rollout 实现。
3. 带高效 AllGather epilogue 的列并行 o_proj 与 LM head。
   NCCL 回退方案正确但更慢。
4. 仅当无法依赖 vLLM 内置 FA3 时，才需要公共确定性分页注意力 kernel。

RMSNorm、row softmax 与 log-softmax 都是小型 Triton kernels，只有在
无法复用 vLLM deterministic kernels 时才应重新实现。

## 5. 任何替代方案必须通过的门禁

每个提议的替代方案都必须通过：

1. 覆盖 prefill 与 decode M 值的算子微测试
2. TP4 分片重建测试
3. 路由器 expert-ID 相等性，而不仅是 logit 容差
4. 每个解码器层的阶段转储
5. KV 长度 127、128、129 与 130
6. 提示词加单条响应，长度 1024
7. 非零优化器更新前后的权重重新加载
8. 重复重新加载与休眠/唤醒周期

数值接近并不足够。验收要求算子边界处 `torch.equal`，
且 K3 严格为零。

## 6. 当前限制与未启用路径

- 已签核启动器使用 `enforce_eager=True`。
  保留 CUDA-graph 预热 hooks 与
  graph-safe 屏障，但完整图捕获不属于
  已签核的 1024-token 结果。
- multimem 实现假设存在支持 NVLS 的互连结构。当 NVLS 不可用时，
  可移植版本应选择 NCCL 或 TMA 回退方案。
- `UNIMATCH_GEMV_MAX_M` 与
  `UNIMATCH_GROUPED_GEMV_MAX_TOKENS`
  均未设置，因此使用内置 dense 与 grouped
  阈值。
- Megatron/Transformer-Engine 路由器、Megatron 对数概率、FP32
  激活与 FC2 前路由权重路径处于未启用状态，因为
  `UNIMATCH_MEGATRON=0`。
- 平坦的 `moe_reduce` 参考实现处于未启用状态；当前合并实现为
  `MoeCombineEpOrdered`。
- 共享模块中存在 Dense Qwen3 MLP 双列代码，但
  Qwen3-MoE decoder 并不使用它。
- 阶段转储 hooks 仅在设置 `UNIMATCH_STAGE_DUMP_DIR` 时启用。

## 7. VeOmni EP4 + DeepEP-HT 扩展

### 7.1 验证中的已签核拓扑

增量 VeOmni 路径使用：

- 物理 GPUs 4–7
- VeOmni 全局规模 4、数据分派规模 4、EP 规模 4
- EP_FSDP 规模 1，每个 rank 32 个本地专家
- 真实 DeepEP 高吞吐量分派/合并
- 四个同机部署的 vLLM TP1/EP1 rollout 实例
- rollout 上设置 `UNIMATCH_TRAIN_EP_SIZE=4`，使 TP1 推理复现
  actor 的 EP4 括号式合并
- BF16、单节点、同步 DeepEP 完成、即时执行

原生 UniRL `deepep_ht` 后端仍可用作 A/B 稳定性
基线。严格路径选择 `ep_comm_backend=unimatch` 与 VeOmni OpSlot：

- `uexact_deepep_ht` 用于 MoE 专家
- `unimatch_bi` 用于 RMSNorm
- `unimatch_vllm_exact` 用于 RoPE
- FA3 用于注意力

除非设置 `UNIMATCH_VEOMNI_EP4_EXACT=1`，当前 FSDP/TP4 路径仍为默认路径。

### 7.2 DeepEP HT 分派

当前 VeOmni 侧：

- `unimatch.adaptor.veomni.dispatcher.DeepEPDispatcher` 为每个 EP 组/设备/模型形状
  缓存一个真实 `deep_ep.Buffer`。
- `get_dispatch_layout` 使用全局 expert IDs 并生成 rank 归属
  元数据。
- `Buffer.dispatch` 将 BF16 隐藏 rows 与 FP32 路由权重传输至
  拥有对应 expert 的 ranks。
- 已接受的约定为 EP4、128 个全局专家、32 个本地专家、top-k 8、
  单节点 NVLink 传输，以及 HT 而非低延迟模式。
- `_DeepEPDispatch` 是自定义自动求导边界。其反向传播使用保存的
  handle，将 hidden 与路由权重梯度合并回源 ranks。

当前 vLLM 侧：

- vLLM TP1 拥有全部本地 experts，不执行 DeepEP 通信。
- 通过匹配分派后专家算术并
  模拟 actor 的 EP4 最终合并结构来获得相等性。

一致原因：

- 分派是置换/通信运算。它必须精确保留全局
  token、slot、expert、routing-weight 与源 rank 标识。
- `canonical_dispatch` 阶段转储对这些标识排序，不依赖
  DeepEP 传输顺序。

公共替代方案：

- 首选 C：核验许可证、构建可复现性与 Torch/CUDA ABI 后，
  依赖 DeepEP 公共 HT API。
- 正确性优先 B：使用 `torch.distributed.all_to_all_single` 并显式携带
  token 元数据。该方案较慢，但可以保留算术，因为 transport
  只复制 rows。
- 不能仅因 UniMatch 本地计算为私有，就认为必须独立重实现 DeepEP。

调试阶段：

- `topk`
- `routing_weights`
- `canonical_dispatch`

### 7.3 rank 本地分组 FC1

当前 VeOmni 侧：

- DeepEP 本地 expert IDs 使用 `-1` 表示无效/非本地 slots。
- 有效 rows 按本地 expert ID 稳定排序，并按按专家填充。
- `LocalGroupedExpertsRuntime` 执行固定
  `(BLOCK_M, BLOCK_N)=(64,128)` 的 TMA grouped FC1。
- `hf_eager` 门控 epilogue 按已签核顺序执行 gate/up GEMM、BF16 实体化、
  BF16 SiLU 实体化与 BF16 乘法。
- 单元素 TP group 保留与 rollout 相同的 TP1 kernel 入口与 epilogue，
  同时让其 AllGather 成为本地存储。

当前 vLLM TP1 侧：

- `UNIMATCH_GROUPED_VARIANT=tma` 避免单元素 NVLS/multimem。
- 分组 runner 为全部 128 个本地专家执行匹配的
  固定顺序 gate/up 路径。

自动求导：

- 精确前向传播使用 UniMatch 分组 kernel，并分离前向传播输入。
- 反向传播通过 `EPMergedFc1GroupGemm` 重新计算
  可微 VeOmni 分组图。

公共替代方案：

- 正确性优先 B：稳定的按专家循环调用同一个公共
  批次不变 linear provider，随后执行即时 BF16 SwiGLU。
- 生产级 C/D：实用性能需要固定调度分组 Triton/CuTe GEMM。
  真实 row 的 K 累加与 BF16 激活边界必须匹配；
  原生自适应 FusedMoE 不能直接替代。

调试阶段：

- `moe_input`
- `fc1`
- `activation`

### 7.4 rank 本地 FC2 与路由权重融合

当前 VeOmni 侧：

- TMA grouped FC2 使用固定 tile `(64,128)`。
- FP32 路由权重在 FC2 epilogue 中相乘。
- 每个加权专家 slot 输出会在 rank 本地
  累加前实体化为 BF16。

当前 vLLM TP1 侧：

- 精确分组路径使用相同的 FC2/路由权重放置位置。
- 最终合并期间不得再次乘以路由权重。

一致原因：

- 在 FC2 BF16 实体化前后移动路由权重乘法
  会改变舍入，并可能改变所选 token 的 log-probabilities。

公共替代方案：

- 正确性优先 B：使用公共 batch-invariant linear 运行 FC2，随后
  执行 `fc2.float() * routing_weight` 并显式转换为 BF16。
- 生产级 C/D：实现带固定 FP32 乘法/BF16
  epilogue 的 grouped GEMM。除非证明这一确切边界，否则必须拒绝
  原生 vLLM FusedMoE。

调试阶段：

- `fc2`

### 7.5 rank 本地 partial 与 DeepEP 合并

当前 VeOmni 侧：

- 专家 slots 恢复为原始 token/slot 顺序。
- 每个 EP rank 以 FP32 累加其拥有的 slots，并将 rank partial 转换为
  BF16。
- `Buffer.combine` 将 rank partials 返回源 ranks。
- `_DeepEPCombine.backward` 使用保存的 handle 反转 communication。

当前 vLLM TP1 侧：

- `MoeCombineEpOrdered(sim_ep_size=4)` 模拟：
  1. 在每个模拟 EP rank 内执行 slot 顺序 FP32 累加
  2. BF16 rank 边界舍入
  3. 按 rank 升序执行 FP32 累加
  4. 末端 BF16 输出

公共替代方案：

- 正确性优先 B：在 EP rank 与 slot 上执行显式 Python/PyTorch loops，
  并插入 BF16 rank 边界转换。平坦的 `torch.sum` 不等价。
- 生产级 C：小型固定顺序 Triton 合并 kernel。
- 原生 EP1 `moe_sum` 如果没有外层 bracketed loop，便无法表示 EP4 rank 边界。

调试阶段：

- `rank_partial`
- `final`

### 7.6 EP4 权重加载、导出与四次 TP1 重新加载

当前 actor 侧：

- HF checkpoint 的按专家划分的 gate/up/down 张量被转换为
  VeOmni 融合 `gate_up_proj` 与 `down_proj` 块。
- EP 分片加载只读取每个 rank 连续的 32 个专家的范围。
- `FullWeightSync._iter_full_tensors_ep`
  AllGather 全部四个本地块，并
  发出规范 HF 按专家名称。

当前 rollout 侧：

- 每个 DP rank 拥有一个 TP1 vLLM 引擎。
- 每个 rank 都接收完整的 128 个专家的载荷。
- vLLM 加载器重建精确本地分组布局。
- 可选抽样 SHA256 验证会在每次 sync 后比较全部四个 rollout 副本，
  发现任何不匹配都会以关闭方式失败。

公共替代方案：

- A：这是名称映射、切片、AllGather 与复制逻辑。不需要私有
  算术 kernel。
- 为限制 memory，应逐 layer/bucket 收集完整专家张量。

### 7.7 EP4 精确环境与 ABI

隔离环境为 `/opt/conda/envs/unirl-veomni-bitwise`：

- Python 3.13
- Torch 2.11 + CUDA 13
- Transformers 5.6
- VeOmni 0.1.11
- vLLM 0.22
- 针对此确切 ABI 从源代码重建 DeepEP

仅导入较旧 DeepEP 扩展并不足够：仅构造函数冒烟测试可能
通过，但 `get_dispatch_layout` 会返回损坏的 CUDA event 元数据。有效的
环境门禁必须执行 dispatch、combine 与反向 handle 复用。

### 7.8 EP 替代/签核门禁

任何公共替代方案都必须通过：

1. 路由器 top-k 与路由权重精确性
2. 本地 TP1 分组 GEMM
3. BF16 中间激活边界
4. rank partial 排序
5. 真实 DeepEP 与 EP 顺序模拟的一致性
6. 通信反向传播
7. EP4 全量权重导出与四个 TP1 摘要相等
8. 所选 token 对数概率 `torch.equal`、absdiff 为零且 K3 为零
9. KV 长度 127–130 与响应长度 1024
10. 更新后重新加载、5-step 与 20-wave 门禁

最终结论必须区分算子门禁完成与完整模型
1024 响应及更新后签核。

### 7.9 已完成的验证证据

环境与算子门禁：

- 针对 Python 3.13、Torch 2.11、CUDA 13 与 SM90，
  从 `/root/deepep_crafts/trmt-deepep` 重建了 DeepEP。
- 已安装 DeepEP 版本：`1.0.0+9c65b0d`。
- `deep_ep_cpp` SHA256：
  `b86b9ea0f34defed11dc9f05b92afed836991af2567846b9b0875a32bf95079f`。
- 真实 EP4 dispatch/combine 与反向 handle 复用已通过。
- 全部六个 UniMatch VeOmni 算子套件均通过，且
  `UNIMATCH_VEOMNI_OPERATOR_GATES_PASS=True`。

历史 decode-topology 完整模型门禁（已被 2026-09-07 整段 forward 门禁取代）：

- 响应 1：K3=0，所选 token 对数概率不匹配数为零
- 响应 8：K3=0
- 响应 32：K3=0
- 响应 128：K3=0，包括 KV 128/129 边界
- 响应 1024：全部四个 actor 各 rank 均报告 `mismatch_count=0`；K3=0
- 连续五次 32-token rollout/重新加载步骤：每一步均为 K3=0
- 连续二十次 32-token 休眠/唤醒轮次：每一轮均为 K3=0
- EP 更改后的原始 HF-FSDP/vLLM-TP4 单 token 回归：K3=0

历史 decode-topology 非零更新与更新后 1024 门禁：

- 第一次 rollout：响应 1024、每个提示词四个样本、奖励 0.4375、
  损失 0.0014、梯度范数 2.4554、K3=0
- 更新后摘要：
  `279f935119b1cfc11fe10e965fd74e654adc8a5fb77878e2af65c6aa32686447`
- 全部四个 TP1 rollout 副本报告相同的 435-parameter 摘要
- 重新加载后第二次 rollout：响应 1024、奖励 0.4375、损失 0.0018、
  梯度范数 2.3926、`mismatch_count=0`、K3=0

主要日志：

- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260829_150359.log`
  — 初始 1024 精确门禁
- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260829_221726.log`
  — 非零更新与更新后 1024 精确门禁
- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260829_231900.log`
  — 五步重新加载稳定性
- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260829_234903.log`
  — 二十轮休眠/唤醒稳定性

### 7.10 2026-09-07 整段 old-policy forward 重新签核

修订后的 alignment probe 不再调用 `decode_topology_replay`。它显式调用
`Qwen3ARStage.old_policy_replay()`，对 rollout 已生成的固定 response tokens
执行与训练 actor 同族的 packed/padded 整段 teacher-forcing forward，并在
FP32 下比较 rollout logprob 与重算 old logprob。

每个 actor rank 的成功条件同时包括：

- shape 和 token count 相同
- 两侧均转换至 FP32，且所有值 finite
- `torch.equal(old_log_probs, rollout_log_probs)`
- `mismatch_count=0`
- `old_rollout_logp_absdiff_max=0`
- `old_rollout_k3_mean=0`
- `old_rollout_k3_max=0`

重新签核结果：

- FSDP/TP4，response=1：四 rank 全部严格为零
- FSDP/TP4，response=128：四 rank 全部严格为零
- FSDP/TP4，response=1024：四 rank 全部严格为零
- VeOmni EP4/vLLM TP1×4，response=1：四 rank 全部严格为零
- VeOmni EP4/vLLM TP1×4，response=1024：四 rank 全部严格为零
- VeOmni EP4 非零 optimizer update 前后：每 rank 4×1024 tokens；
  更新前和重新加载后的更新后门禁均全部严格为零
- 更新后四个 TP1 rollout 副本的 435-parameter digest 仍一致：
  `279f935119b1cfc11fe10e965fd74e654adc8a5fb77878e2af65c6aa32686447`

新日志：

- `logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_173244.log`
  — FSDP/TP4 response=1
- `logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_174006.log`
  — FSDP/TP4 response=128
- `logs/qwen3_moe_fsdp_vllm_tp4_bitwise_20260907_174522.log`
  — FSDP/TP4 response=1024
- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_175208.log`
  — VeOmni EP4 response=1
- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_181046.log`
  — VeOmni EP4 response=1024
- `logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_20260907_181755.log`
  — 非零更新与更新后 4×1024 整段 old-logprob 门禁

所有新门禁均在最终 logprob 层直接通过，因此本轮没有产生失败 token，
也无需向
`/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/tensor_dump`
写入逐层差异 tensor。训练用 grad-enabled replay 的
`train|Δlogp|mean` 仍单独记录，不与 old-policy 门禁的指标混用。

### 7.11 现已设防的集成故障

- VeOmni 0.1.11 的通用融合 MoE adapter 将 FP32 top-k 权重转换为 BF16。
  现在精确 sparse-block bridge 会将路由器保存的 FP32 权重
  直接传给绑定的 UniMatch 专家 OpSlot。
- TP1 vLLM 比 TP4 需要更高的模型预算，但不受约束的缓存
  会挤占更新后 actor 状态。已签核配置固定使用 1 GiB KV cache。
- 布尔休眠状态无法表示仅权重唤醒。直接 vLLM
  引擎现在跟踪部分唤醒，并分别映射 `weights`，随后映射 `kv_cache`。
- EP4 提取在 actor 卸载前完整落到 CPU；只有此后才会映射 vLLM
  权重并推送 buckets。
- 在 vLLM CuMem 休眠前关闭 UniMatch TMA/scratch 缓存，避免陈旧
  设备地址在重新映射后继续存在。
- VeOmni 0.1.11 梯度裁剪曾一次性以 FP32 实体化所有梯度。已签核
  启动器使用 EP-aware 分块流式 norm 与原地缩放。
- 此 kernel 上的 CUDA IPC 无法使用
  `expandable_segments=True`；启动器
  显式保持禁用该项。
