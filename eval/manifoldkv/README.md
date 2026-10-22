# ManifoldKV implementation (execution on hold)

Base: `main` at `7d14d001c9e70014de515dbd00ee6986d50d365f`; independent branch `manifoldkv`.

Paper: [ManifoldKV](https://arxiv.org/html/2602.08343v1). Algorithms 1/2 and Appendix B are the implementation reference.

Each KV head ranks raw post-RoPE keys by Euclidean distance from their
centroid and retains `max(1, floor(keep_ratio * prompt_length))` tokens.
`keep_ratio` is the retained fraction; the paper's rho is the removed fraction.
No sink/recent-window reservation is added. `centroid_window=0` is Algorithm 1;
`4096` enables Algorithm 2's non-overlapping local centroids followed by global
top-k. This is the standalone scorer, without AdaKV's variable head budgets.

The source is Algorithms 1/2 and Appendix B of the pinned paper. Its public
text identifies the implementation as anonymous; no author repository was
identified. This port is paper-derived rather than a verified author-code port.
The one-token minimum defines the small-prompt boundary where floor gives zero.

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
