# StreamingLLM

This branch implements the rolling cache with attention sinks from
[Efficient Streaming Language Models with Attention Sinks, §3.2](https://arxiv.org/html/2309.17453v4).
The method is independent of `origin/snapkv`: it uses no attention scoring,
pooling or top-k selection. Between attention steps it keeps the first sink
tokens and the latest recent tokens. The next query attends to that retained
past plus itself, then the oldest recent token is evicted.

Add this optional object to a Qwen3 CausalLM `nntr_config.json`:

```json
"streaming_llm": {"sink_tokens": 4, "recent_tokens": 252}
```

Omitting the object preserves the normal dense attention path. An empty object
uses 4 + 252. Both numbers must be unsigned integers; `recent_tokens` must be
positive. Set `sink_tokens` to 0 for a window-attention ablation. The sum must fit
`max_seq_len` and the model's positional table.

Run the usual CausalLM executable with `NNTR_ENGINE=cpu`. Support currently covers
batch-one Qwen3 with FP32 or FP16 activations and the existing half-precision KV
storage. Training, GPU inference, learned attention-sink weights, sliding-window
models, int8 KV, FSU, SnapKV, saved KV caches and `skip_prefill` are rejected.
Quantized projection weights continue to use the existing CPU layers.

The physical K/V allocations have `sink_tokens + recent_tokens` rows in each
layer, rather than `max_seq_len` rows. Keys are stored before RoPE. Attention
rotates a temporary key view at contiguous positions starting at zero, and
rotates queries at their current cache positions. Values retain their original
order. Before the cache fills, attention can process a dense block; after it
fills, each query attends before evicting just the oldest recent token. This
also handles prefill chunks crossing the capacity without removing keys needed
by their early queries. Attention scratch and the RoPE table include one extra
row for the incoming token, so a full 4 + 252 cache attends over 257 tokens at
positions 0 through 256. Persistent KV storage remains 256 rows. Attention
scratch is bounded by cache capacity and prefill plane.

Prompts are processed fully in `init_seq_len` chunks even when they exceed
`max_seq_len` or `max_position_embeddings`. Tokenization and output text still
use memory proportional to the submitted prompt and returned text. Explicit
`num_to_generate` can exceed the cache size; an absent or non-positive limit
uses `max_seq_len` as the per-call generation budget. EOS and cancellation still
stop generation. All emitted tokens, including the final token, are committed
to the cache so a later `run()` can append another turn. Use
`setSessionTokenPosition(0)` to start a new stream. Other arbitrary repositioning
is rejected; the absolute stream counter is checked before addition.

StreamingLLM cannot retrieve evicted context. This implementation does not train
a new sink token or extend the model's retained attention window.

## Paper fidelity audit

The cache behavior follows §3.2 and the authors' token-by-token
[perplexity driver](https://github.com/mit-han-lab/streaming-llm/blob/2e5042606d69933d88fbf909bd77907456b9b4dd/examples/eval_long_ppl.py),
which trims the returned cache **after** computing attention. Their
[cache policy](https://github.com/mit-han-lab/streaming-llm/blob/2e5042606d69933d88fbf909bd77907456b9b4dd/streaming_llm/kv_cache.py)
also exposes `evict_for_space` for reserving an entire generation turn; this
implementation uses the token-by-token policy consistently for prefill and
decode, independent of chunk size.

| Requirement | Implementation and regression coverage |
| --- | --- |
| Keep the initial sink KV without recomputing past hidden states | Raw cached keys/values are checked against the original token projections after each step; repeated eviction preserves the sinks. |
| Keep the newest recent KV in chronological order | Attention runs before eviction; the returned cache retains exactly the first sinks and last recent tokens. |
| Use cache positions rather than original text positions | Queries use the retained past length; keys use contiguous scratch positions. The scalar RoPE/softmax oracle covers streams beyond the model's positional limit. |
| Store keys before RoPE and rotate them for each attention step | Tests check raw key bits after each forward, including eviction. Rotated keys exist only in temporary scratch. |
| Preserve the paper's boundary behavior | The Figure 4 regression uses past `[0,1,2,3,6,7,8]` plus current token `9`, at positions `0..7`. A separate regression fills the whole configured position window. |
| Bound cache memory across long streams | Each layer allocates only `sink_tokens + recent_tokens` persistent rows; model tests cover long prefill, decode, continuation and reset. |

An earlier version evicted before attention and therefore attended to one fewer
past token once full. That boundary difference is corrected. This is an
inference implementation for Qwen3 with RoPE; it does not implement ALiBi model
families or the separate sink-token pre-training experiment in §3.3. It uses
the repository's half-precision KV storage and CPU attention arithmetic, so
bitwise equivalence with the authors' Hugging Face models is not claimed.
Pretrained PG-19 perplexity and the paper's performance results have not been
reproduced. The absolute stream counter currently uses the repository's unsigned
32-bit inference interface and rejects overflow.

## Evaluation

The evaluation follows the latency/RSS protocol from `origin/snapkv`, using
default input lengths 512/1024/2048, batch one, maximum sequence setting 4225,
128 requested decode steps, four NNTrainer/OMP threads, one OpenBLAS thread,
five latency and five RSS
trials per condition, alternating condition order and a one-second interval.
Generation sampling and EOS settings are preserved. Prompts repeat token 5555;
the tokenizer must verify their exact length. Override the token ID or lengths
when evaluating a different tokenizer.

The maximum sequence setting reserves one more row than SnapKV's 4224: the
current dense driver starts its decode loop one row after the prompt. With
4224 rows, a 4096-token prompt supplied with `--contexts 4096` would allow only
127 decode forwards. The extra row lets both conditions perform all 128
requested forwards at that input length.

Install the Python `tokenizers` package, build the branch with transformer
support, and use a plain model configuration without chat/system prompts:

```bash
./eval/streaming_llm_eval.sh /path/to/qwen3-model \
  --binary "$PWD/build/Applications/CausalLM/nntr_causallm" \
  --cpus 0,2,4,6 --check

./eval/streaming_llm_eval.sh /path/to/qwen3-model \
  --binary "$PWD/build/Applications/CausalLM/nntr_causallm" \
  --cpus 0,2,4,6
```

The same executable runs dense and StreamingLLM conditions by default. A separate
baseline can be supplied with `--baseline-bin`; use matching build options and
runtime libraries. Model JSON files are staged in a temporary directory and
asset paths are made absolute; the original files are preserved.

Latency trials execute the model directly. Memory trials use the unchanged
`Applications/utils/mem_usage.sh` and require positive target-process RSS samples
consistent with its reported peak. A run too short for that script's one-second
sampling period fails validation. The OS-reported peak is recorded separately.

For a deployed Android CPU executable and identical local/device model assets:

```bash
./eval/android_streaming_llm_eval.sh /path/to/local/qwen3-model \
  --binary /data/local/tmp/nntrainer/causallm/nntr_causallm \
  --device-model /data/local/tmp/nntrainer/causallm/models/qwen3 \
  --serial DEVICE_SERIAL --cpus 0,2,4,6 --check
```

Omit `--check` to measure. Android uses a unique staging directory, verifies
device asset hashes and cools to thermal_zone0 ≤ 40000 millidegrees before each
trial. Temperature read failures or a five-minute cooling timeout stop the run.
Paths and thresholds are configurable. Deploy matching shared libraries beside
the executable. Android execution requires a connected device and has not been
validated by the host-only tests.

Results go to `logs/streaming_llm/<timestamp>`: raw logs, CSV, a JSON manifest
with configurations, commands, asset/binary hashes and environment, and a
mean ± sample standard deviation summary. Use `--output` for another location.
Incomplete or invalid trials are recorded and stop evaluation. This harness
does not resume interrupted evaluations.

The legacy dense driver can emit an additional prefill token outside its
reported decode count. StreamingLLM counts every emitted token and performs a
final cache commit for continuation. Both conditions perform the requested
number of decode forwards when EOS does not intervene, but inspect actual token
counts and early EOS before interpreting fixed-work comparisons. These are
latency/RSS comparisons, not quality scores. Reproducing the paper's PG-19
teacher-forced perplexity evaluation requires an additional scoring driver and
pretrained model/data assets.

## Tests

```bash
ninja -C build Applications/CausalLM/unittest_streaming_llm \
  Applications/CausalLM/unittest_streaming_llm_models
meson test -C build unittest_streaming_llm unittest_streaming_llm_models
python3 -m unittest discover -s eval -p 'test_*.py'
```

The attention tests use an independent scalar RoPE/softmax reference with GQA,
test token-by-token and chunked execution beyond the model's position limit,
check sink preservation over repeated eviction, and include the zero-sink
ablation, a one-token recent window, the paper's Figure 4, and a cache as large
as the model's position window. Model tests exercise bounded allocation, long
prompt/generation, multi-turn continuation, reset and invalid configurations
using tiny models.
