# DropKV implementation and CPU evaluation

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

Implementation status: **Linux and native Windows validation completed; 72/72 measurements; final five Windows memory trials completed on 2026-10-28**.
See [CPU results and limitations](results/20261023-cpu.md) and [individual trials/provenance](results/20261023-cpu.json).
LongBench and quality evaluation were excluded. The final five Windows 4096-token memory trials were resumed with the identical protocol and artifacts on 2026-10-28 after PR creation.

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

Execution history: the three methods were measured sequentially on Linux and native Windows after the user's signal. The pipeline was paused on 2026-10-23 and resumed after all eight PRs were posted on 2026-10-28. Only the five pending Windows 4096-token memory trials were run, with matching helper, protocol, models, binaries and runtime hashes; all 72/72 measurements are now complete.

## Reproduce performance measurements

Build separate main/method release artifacts with matching options first. Use the pinned Qwen3-0.6B model assets recorded in the results JSON. The helper checks build options, model hashes, actual token counts, and policy/allocation markers. Run each platform sequentially, with no overlapping benchmark process.

Linux (add `--check` for a preflight that does not execute inference):

```bash
python3 eval/dropkv/dropkv_eval.py /path/to/qwen3-fp32 \
  --baseline-bin /path/to/main-build/Applications/CausalLM/nntr_causallm \
  --dropkv-bin /path/to/method-build/Applications/CausalLM/nntr_causallm \
  --repeat 3 --max-seq-len 4225 --cpus 0,2,4,6 --output logs/cpu-linux
```

Native Windows PowerShell, from the same branch. The evaluated MinGW builds use `nntr_causallm_eval.exe` with the same original-main-return adapter in main and method. Staging/build scripts are preserved in the recorded run directory.

```powershell
py -3.12 eval/dropkv/dropkv_eval.py C:/path/to/qwen3-fp32 `
  --baseline-bin C:/path/to/main-build/Applications/CausalLM/nntr_causallm_eval.exe `
  --dropkv-bin C:/path/to/method-build/Applications/CausalLM/nntr_causallm_eval.exe `
  --repeat 3 --max-seq-len 4225 --output logs/cpu-windows
```

Use `--resume logs/cpu-linux` (or the Windows directory), without `--output`, only for a matching interrupted run. The helper rejects changed protocols or invalid completed trials. Any native exit adapter must match between main and method, as documented in the results.
