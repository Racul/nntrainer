# VaSE-DKV implementation (execution on hold)

Base: `main` at `7d14d001c9e70014de515dbd00ee6986d50d365f`; independent branch `vase-dkv`.

Paper: [VaSE-DKV](https://arxiv.org/html/2606.03928v1). Official source: [eval/reasoning_tasks/modified/transformers/cache_utils.py](https://github.com/terarachang/VaSE/blob/a7a523d8cde1ff9248925c4978ba8ffe7b3630bc/eval/reasoning_tasks/modified/transformers/cache_utils.py), pinned at `a7a523d8cde1ff9248925c4978ba8ffe7b3630bc`.

This implements the official `cur_resample_gauss` variant, not VaSE-AttnV.
After a decode output, eviction runs when physical length exceeds budget and is
a multiple of buffer. The most recent buffer rows are protected. Historical
scores are `||K_i G||^2 * ||V_i G||^2`, normalized per head; four sinks are
protected. A fresh Gaussian `G` of shape `[head_dim, rank]`, scaled by
`1/sqrt(rank)`, is shared across heads at each eviction. Default rank is 20.
The persistent budget INCLUDES the protected buffer. There is no prefill eviction.

Port choices: chronological gather instead of score order; explicit infinite
sink scores and a deterministic zero-product tie policy; portable seeded
Box-Muller generation instead of PyTorch's generator. Random samples and FP32
projection arithmetic do not reproduce CUDA/BF16 bits. This port requires
budget divisible by buffer and at least four historical slots. Unlike the
author allocator, it releases a long-prompt slab at first eviction and then
keeps only budget+buffer rows. These changes do not replace the score policy.
With a short prompt and short decode, no eviction may occur: report the actual
eviction count, and never label that case as demonstrated eviction speedup.

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
