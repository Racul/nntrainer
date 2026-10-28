# ManifoldKV

NNTrainer implements the training-free geometric prompt scorer from
[ManifoldKV: Training-Free KV Cache Compression via Euclidean Outlier Detection,
ICML 2026](https://proceedings.mlr.press/v306/datta26b.html).

Add this object to a Qwen3 CausalLM `nntr_config.json`:

```json
"manifold_kv": {"budget": 256, "window_size": 0}
```

`budget` is the maximum number of **prompt tokens retained per KV head** in
each layer. It must be positive and fit `max_seq_len`. An empty object uses
256. Omitting the object keeps the normal inference path.

`window_size: 0` uses one centroid over the entire prompt. A positive value
enables WindowedManifoldKV: each consecutive, non-overlapping window has its
own centroid, including the final partial window. For example, use 4096 for
the paper's 4K local-centroid variant. Selection still uses one prompt budget
per KV head; the window size is not a recent-token reservation.

The implementation supports batch-one Qwen3 CPU inference with FP32 or FP16
activations and half-precision KV storage. Existing quantized projection
weights can be used. Set `NNTR_ENGINE=cpu`; GPU inference, training,
sliding-window models, int8 KV, FSU, skipped prefill, other eviction policies,
and saved-cache loading/saving are unsupported and rejected.

The complete prompt, including system and tail text, must fit `init_seq_len`
and leave a generation position within both `max_seq_len` and the model's
RoPE table. An oversized prompt raises an error instead of being truncated.
This is a single dense prefill, followed by prompt compression once in each
layer. Generated tokens are appended without further eviction. Generation
stops at EOS, cancellation, `num_to_generate`, or the original position limit.
Every emitted token, including the final token, is committed to the cache.

Use `setSessionTokenPosition(0)` before submitting another prompt. Continuing
a compressed prompt with a new multi-token turn is not implemented.

## Scoring and paper fidelity

For each layer and KV head, compute the FP32 centroid of the cached, rotated
prompt keys and rank tokens by squared Euclidean distance to that centroid.
Squared distance preserves the ordering of the paper's L2 score and avoids
square roots. Equal scores prefer earlier token positions deterministically.
Selected tokens are copied in chronological order, and K/V head slices remain
paired. Query heads in a GQA group share their KV head's selection.

Prompt attention is computed densely before eviction, preserving the prompt
hidden states passed to subsequent layers. Keys retain their original RoPE
encoding after compaction. Decode queries and incoming keys use original
logical token positions; physical cache positions advance independently.
There is no key renumbering, attention scoring, pooling, sink protection,
recent-token carve-out, learned scorer, or stochastic sampling.

This implementation uses a fixed budget for each KV head. It does not
reproduce the separate AdaKV adaptive head-budget framework used in some of
the paper's experiments. Applying the scorer repeatedly during streaming
would be a different policy and is not implemented.

## Memory and validation

With a positive `num_to_generate`, each persistent cache plane has
`min(max_seq_len, budget + num_to_generate)` rows, rather than `max_seq_len`.
If the generation limit is absent or non-positive, the cache reserves the
whole logical sequence allowance. Full prompt K/V temporarily exist for the
layer currently running prefill; they are released after that layer's
selection. Prompt activations and temporary attention buffers also contribute
to peak memory, so the persistent KV ratio is not an end-to-end RSS ratio.

The inference log reports the retained prompt count and allocated cache rows,
as well as the usual prefill, generation, latency, and peak-memory metrics.

```bash
ninja -C build Applications/CausalLM/unittest_manifold_kv \
  Applications/CausalLM/unittest_manifold_kv_models
meson test -C build unittest_manifold_kv unittest_manifold_kv_models
```

Tests compare global/windowed selection and GQA attention against independent
scalar calculations, check original RoPE positions and selected K/V bytes,
and exercise allocation, reset, FP16 activations, dense-prefill equivalence,
and unsupported configurations. Published quality and speed results have not
been reproduced in NNTrainer. Evaluate real prompts on retrieval and
generation tasks alongside latency and memory; repeated-token timing prompts
alone do not establish quality.

## Latency and memory evaluation

Install Python's `tokenizers` package and build the CausalLM executable. The
standalone runner compares dense attention and ManifoldKV using the same
binary, weights and generation settings:

```bash
./eval/manifold_kv_eval.sh /path/to/qwen3-model \
  --binary "$PWD/build/Applications/CausalLM/nntr_causallm" \
  --cpus 0,2,4,6 --check

./eval/manifold_kv_eval.sh /path/to/qwen3-model \
  --binary "$PWD/build/Applications/CausalLM/nntr_causallm" \
  --contexts 512 1024 2048 --budget 256 --decode 128 --repeat 5 \
  --cpus 0,2,4,6
```

The default protocol runs five latency trials and five separate peak-RSS
trials per condition, alternates condition order, and uses four NNTrainer/OMP
threads and one OpenBLAS thread. Prompts repeat token 5555; preflight checks
their exact tokenized length. Use a plain model configuration without chat or
system prompts. Model JSON is staged temporarily and asset paths are made
absolute, preserving the supplied model directory.

Results are saved under `logs/manifold_kv/<timestamp>`: raw logs, CSV, a JSON
manifest, and `summary.md` with latency in seconds, peak RSS in MiB, percentage
changes, and sample standard deviations. Use `--output` for another location.
Latency is the reported prefill-plus-generation time. RSS trials use
`Applications/utils/mem_usage.sh`; invalid measurements stop evaluation.
These repeated-token trials do not measure output quality. Android CPU
evaluation is also available through `--platform android`, `--device-model`
and `--serial`; it has not been validated on a device.
