# PyramidKV CPU implementation and evaluation

This branch starts at main `7d14d001c9e70014de515dbd00ee6986d50d365f`.
The policy follows [PyramidKV](https://arxiv.org/abs/2406.02069) and the
official `PyramidKVCluster` implementation at revision
`68cd9551a63fedf362ddb0edb008badd04c24996` of
[KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory).

Set the following optional field in `nntr_config.json`:

```json
"pyramidkv": {"budget": 256, "window": 64, "kernel": 5, "beta": 20}
```

Omitting the field or using budget zero preserves the existing path. Active
support is Qwen3 FP32 weights/activations, FP32 or UINT16 (FP16-bit) KV storage,
CPU, batch one, one fresh full prefill and consecutive single-token decode.
Cached/skipped/chunked prefill, resumed sessions, serialized caches, training,
sliding windows, sinks, ring caches, int8 KV, nonzero attention softcapping,
FP16 activations and device engines are rejected when enabled.

Prefill attention uses the original kernel. After its output is complete,
the last `window` causal softmax rows vote on older keys. Query heads sharing
one KV head are averaged, followed by zero-padded average pooling of width
`kernel` and independent top-k selection per KV head. All recent window tokens
are retained. Absolute RoPE positions remain unchanged; selected K/V move
together in chronological order. All decode tokens are subsequently appended;
there is no H2O-style online eviction.

For `P = budget - window`, the official integer schedule is
`low = P // beta`, `high = 2*P - low`,
`step = (high - low) // (layers - 1)`, and
`retained[layer] = high - layer*step + window`. `high` is clipped to the
available prefix, with `low = 2*P-high`. Prompts shorter than `budget` remain
intact; prompts shorter than `2*P` use the uniform budget. A one-layer model
uses the uniform budget. With 28 layers and the example settings, a long
prompt retains 439 entries at layer 0 and 88 at layer 27, mean 263.5. Integer
rounding is intentional. Equal-score ties prefer earlier tokens.

The shared KV-head GQA policy corresponds to the official implementation's
`kv_head` layout with mean aggregation. It differs from the paper's original
expanded query-head layout. Sorting selected positions changes their physical
order relative to official score-order top-k, while preserving their content
and absolute positions. Existing full-capacity allocations are retained:
this reduces decode attention span, not allocated KV payload. Only observation
rows are materialized for scoring, avoiding a new full triangular score tensor.
CPU decode uses the existing per-row QK/softmax/AV kernels over compact views.

## Latency and sampled peak memory

Use `python3 eval/pyramidkv/pyramidkv_eval.py --help` and `--check` first.
Native Windows uses the same Python entrypoint with Windows Python and `.exe`
paths. The helper derives from the existing H2O measurement code, with
PyramidKV-specific settings and validation of every layer's retention marker.

```bash
python3 eval/pyramidkv/pyramidkv_eval.py models/qwen3-fp32 \
  --baseline-bin /path/to/main/build/Applications/CausalLM/nntr_causallm \
  --pyramidkv-bin /path/to/pyramidkv/build/Applications/CausalLM/nntr_causallm \
  --cpus 0,2,4,6 --repeat 3 --output /path/to/new-results
```

Defaults: exact token-5555 prompts of 512/2048/4096 tokens, at most 128 decode
steps, capacity 4225, batch one, NNTR/OMP four threads and OpenBLAS one thread.
Capacity 4225 allows 128 decode steps even at input 4096 in the existing loop.
Sampling and EOS settings remain those of the model. Function runs are excluded
from statistics. Each condition/input has three latency and three memory
processes (36 measured trials per platform); condition order alternates.

Latency is CLI `total:` excluding loading. Linux memory is the maximum sampled
model-PID smaps RSS from the original `mem_usage.sh`. Native Windows memory is
sampled current Working Set via PSAPI at 50 ms. OS peaks are separate. These
metrics include loading/prefill and are not dedicated KV allocation sizes.
`--resume OUTPUT` reuses only matching valid completed trials. Invalid trials
remain recorded and stop the controller. Source/build provenance is recorded
separately; a checkout HEAD alone does not identify an executable.

## LongBench v1

`longbench_eval.py` evaluates all 21 tasks and 4,750 samples without subsampling.
It requires a local copy of the official dataset and scoring sources; it does
not download code or data implicitly. Use the same pinned sources on both
platforms and compare the saved protocol hashes before resuming.

This run pins the official [LongBench source](https://github.com/THUDM/LongBench)
at `2e00731f8d0bff23dc4325161044d0ed8af94c1e` and the
[dataset](https://huggingface.co/datasets/THUDM/LongBench)
at `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`.
`--official` points to the directory containing `eval.py`, `metrics.py` and
`config/`; `--data` points to the extracted v1 JSONL directory, not LongBench-E.

The full evaluation uses official task prompts, generation limits, metrics,
middle truncation and greedy generation. The context limit is explicit in
the command and recorded per sample. This is a Qwen3-0.6B port evaluation,
not a reproduction of the paper's larger-model scores. Quality predictions
are generated once per condition/platform; three-repetition performance trials
above are a separate experiment. Partial scores are always labelled partial.

```bash
python3 eval/pyramidkv/longbench_eval.py --prepare \
  --model models/qwen3-fp32 --official /path/to/pinned/LongBench \
  --data /path/to/data --cases /path/to/cases-8192.jsonl --max-input 8192

python3 eval/pyramidkv/longbench_eval.py \
  --model models/qwen3-fp32 --official /path/to/pinned/LongBench \
  --cases /path/to/cases-8192.jsonl --max-input 8192 \
  --baseline-bin /path/to/main/nntr_causallm \
  --pyramidkv-bin /path/to/pyramidkv/nntr_causallm \
  --output /path/to/full-longbench-results
```

The official scorer needs `rouge`, `jieba`, `fuzzywuzzy` and NumPy; case
preparation also needs `tokenizers` and Jinja2. Existing cases and records are
never replaced silently. Interrupted model logs are retained under a timestamp
before a retry. Resume rechecks the full protocol, binary/library/model/case
hashes and valid prediction records. Samples alternate condition order, and
the manifest reaches COMPLETE only after every expected prediction is valid.

NNTrainer's prefill can produce the first new token separately from the decode
loop. Quality evaluation sets `init_seq_len = actual_prompt_tokens + 1` to
register that first output, and requests `official_max_new_tokens - 1` decode
steps. This is checked with a tiny deterministic model. Latency measurements
above retain their original protocol. Qwen3 uses its official non-thinking chat
template except for the six tasks that official LongBench runs without chat
wrapping. SAMSum adds newline to EOS, as in the official prediction script.
