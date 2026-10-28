# ManifoldKV implementation and CPU evaluation

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

Implementation status: **Linux and native Windows validation completed on 2026-10-23; 72/72 measurements**.
See [CPU results and limitations](results/20261023-cpu.md) and [individual trials/provenance](results/20261023-cpu.json).
LongBench and quality evaluation were excluded. Publication did not resume measurements.

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
freed pages. Measured changes and their limits are reported in the linked CPU results.

`settings.json` is a patch to a complete nntr config, not a standalone model
pack. C++ tests are registered in `unittest_causallm_models`: analytic score
cases, real Tensor K/V gather and owner release, absolute-position boundaries,
disabled/unpruned tiny-model comparisons, activation, repeated decode, and
unsupported settings. The recorded Linux and native Windows selected suites passed; see the linked CPU results.

Execution history: the three methods were measured sequentially on Linux and native Windows after the user's signal. The pipeline was stopped at the user's request with DropKV Windows memory incomplete. No new inference or benchmarks were run for publication. Use the result's completed counts rather than assuming a complete sweep.

## Reproduce performance measurements

Build separate main/method release artifacts with matching options first. Use the pinned Qwen3-0.6B model assets recorded in the results JSON. The helper checks build options, model hashes, actual token counts, and policy/allocation markers. Run each platform sequentially, with no overlapping benchmark process.

Linux (add `--check` for a preflight that does not execute inference):

```bash
python3 eval/manifoldkv/manifoldkv_eval.py /path/to/qwen3-fp32 \
  --baseline-bin /path/to/main-build/Applications/CausalLM/nntr_causallm \
  --manifoldkv-bin /path/to/method-build/Applications/CausalLM/nntr_causallm \
  --repeat 3 --max-seq-len 4225 --cpus 0,2,4,6 --output logs/cpu-linux
```

Native Windows PowerShell, from the same branch. The evaluated MinGW builds use `nntr_causallm_eval.exe` with the same original-main-return adapter in main and method. Staging/build scripts are preserved in the recorded run directory.

```powershell
py -3.12 eval/manifoldkv/manifoldkv_eval.py C:/path/to/qwen3-fp32 `
  --baseline-bin C:/path/to/main-build/Applications/CausalLM/nntr_causallm_eval.exe `
  --manifoldkv-bin C:/path/to/method-build/Applications/CausalLM/nntr_causallm_eval.exe `
  --repeat 3 --max-seq-len 4225 --output logs/cpu-windows
```

Use `--resume logs/cpu-linux` (or the Windows directory), without `--output`, only for a matching interrupted run. The helper rejects changed protocols or invalid completed trials. Any native exit adapter must match between main and method, as documented in the results.
