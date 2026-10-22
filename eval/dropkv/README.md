# DropKV implementation (execution on hold)

Base: `main` at `7d14d001c9e70014de515dbd00ee6986d50d365f`; independent branch `dropkv`.

Paper: [DropKV](https://arxiv.org/html/2610.22130v1). Official source: [dropkv_cache.py](https://github.com/AozhongZhang/DropKV/blob/afb97c2af09418ca58399dba50e6f956bb8ef153/dropkv_cache.py), pinned at `afb97c2af09418ca58399dba50e6f956bb8ef153`.

For every query head in a KV group, tail-window queries receive their original
causal mask. Scores sum `(p/(1-p+1e-6))^2 * ||attention_output - V_i||^2`
over window rows, then average across query heads AFTER the nonlinear score.
The recent window is protected, and the highest scores are retained. The target
is `min(N, max(floor(keep_ratio*N), min(window,N)))`. Decode appends without
further eviction.

Paper Algorithm 1 specifies max pooling, while pinned `DropKVCache` uses average
pooling. Both are explicit: default `pooling=max` follows the paper; `avg`
reproduces the standalone pooling choice with zero padding included in its
divisor. These variants must not be combined in one benchmark condition.

The implementation uses FP32 CPU BLAS and half KV, not Triton, CUDA tensor
cores, BF16 probability rounding, or the paper's fused GPU kernels. Values are
converted once per head; norms are reused and residual distances use dot
products. Only one probability row is stored, with no window*N*dim tensor.
This ports the eviction rule, not the reported GPU acceleration.

Implementation status: **written; build, unit tests, inference and benchmarks
NOT RUN**, per the user's execution hold. Formatting and source review do not
establish correctness or performance. No PR/CI was started for this change.

The supported entry point is `CausalLM::run`: Qwen3, FP32 weights/activations,
batch 1, CPU, full causal attention, one fresh full prefill, then single-token
decode with a positive `num_to_generate`. Set `skip_prefill=false`, provision
`init_seq_len` for the entire prompt, and set `max_seq_len` and RoPE limit for
prompt plus generation. Truncation is rejected. GPU, FP16 activations, INT8 KV,
sliding/ring/sink attention, chunking, resumed sessions and KV serialization are
rejected on the enabled path. Missing settings or `enabled=false` preserves
baseline wiring. Reset session position to zero to start another fresh request.

Actual storage is layer-owned FP16/UINT16 half-bit tensors. Graph cache inputs
remain one-row placeholders to preserve input alias topology. After prefill,
attention views expire before a gather replaces the owned tensors. Decode uses
physical `live` for cache offsets and absolute `next` for RoPE. The old full KV
slab has no persistent owner after replacement. Allocation markers count both
K and V (four bytes per token/head dimension), excluding the tiny placeholders.
KVCacheManager itself counts only those placeholders on the enabled path.

A smaller live allocation does not guarantee lower process peak/RSS. During
gather, old/new slabs coexist; full prefill activations, attention scratch,
scoring buffers and weights still contribute to peak. The allocator may retain
freed pages. No memory or latency improvement is claimed before measurement.

`settings.json` is a patch to a complete nntr config, not a standalone model
pack. C++ tests are registered in `unittest_causallm_models`: analytic score
cases, real Tensor K/V gather and owner release, absolute-position boundaries,
disabled/unpruned tiny-model comparisons, activation, repeated decode, and
unsupported settings. They have not been executed.

Execution hold: await an explicit user signal, then run **ManifoldKV → VaSE-DKV
→ DropKV sequentially**, never overlapping methods/builds/measurements on the
same PC. Carry forward Linux and native Windows separately, main baseline and
method three trials per input/phase, latency and memory in separate processes.
Prior candidate inputs are 512/2048/4096 and decode 128; freeze the actual
configs, source hashes, model hashes and activation counts before measuring.
Use the existing OBCache/H2O measurement helpers after adapting the method
settings/markers. LongBench remains excluded. Only after authorized execution
and valid measurements should tables and a PR claim validation results.
