# Optimized parity kernels

The open-source experiment defaults to `public_reference`, implemented with
vLLM, PyTorch, DeepEP and NCCL public APIs.

No kernel from the current external UniMatch worktree is copied here because
that repository declares its license as `TBD`. The `optimized` profile fails
closed until every imported kernel has:

1. an identified author and source revision;
2. an Apache-2.0-compatible license;
3. a written arithmetic contract;
4. operator micro-gates and full 1024-token parity results.

Future experiment-owned implementations belong in the sibling `rmsnorm/`,
`grouped_moe/` and `ep_combine/` packages.
