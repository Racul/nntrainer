# KV-cache method evaluation

`kvcache_eval.py` compares a baseline CausalLM executable with method
executables built from any branch (KV-cache eviction or otherwise) on every
project metric. A new method needs no change to this script:

| Metric | Subcommand | Source |
|---|---|---|
| End-to-end latency, model loading excluded | `perf` | CausalLM `total:` |
| Latency including model loading | `perf` | `[e2e time]:` |
| Peak memory (primary) | `perf` | sampled peak RSS of the model PID from the unmodified `Applications/utils/mem_usage.sh` |
| Prefill / decode throughput | `perf` | `prefill:` / `generation:` token counts and ms |
| OS peak RSS (reference only) | `perf` | `Max Resident Set Size:` (ru_maxrss) |
| Stability | `perf`, `quality` | sample SD over repetitions, invalid-trial and early-EOS counts |
| Output quality guardrail | `quality` | LongBench v1 official metrics |

The script does not build anything. Build each condition from its own branch
with identical Meson options and compiler, then pass the executables. A run
refuses differing builds unless `--allow-build-difference` is given, which
marks the run. All conditions of one run share one model directory, so compare
executables built on the same base.

Convert the model with the converter of that base. Since `5103c857` the Qwen3
converter writes FFN weights as gate, up, down, and main no longer remaps the
old up, gate order (`1829e4de`). A `.bin` made by an older converter loads on
main without error but generates fluent-looking garbage; the validity checks
cannot detect that, so inspect a few outputs of every new setup.

## Conditions

The script holds no list of allowed methods. A condition is described by these options:

| Option | Meaning |
|---|---|
| `--condition LABEL=BINARY` | Repeatable. The label is free (`[A-Za-z0-9._-]`, up to 64 characters). `baseline` is the reference: it runs the model's own `nntr_config.json`, and every other label is compared with it. |
| `--setting LABEL=JSON` | The `nntr_config.json` keys this condition adds, e.g. `'capkv-r50={"capkv": {"ratio": 0.5}}'`. Keys the harness controls cannot be set: prompt, lengths, batch, model files, chat fields. |
| `--expect LABEL=REGEX` | Repeatable, multiline regex. A log line the method prints only while it is active. It must appear in every run of LABEL and must not appear in any baseline run, so it must tell the two apart. |
| `--unchecked LABEL` | Runs a non-preset condition without `--expect`. The summary marks it **UNCHECKED: activation not proven**. |
| `--preset LABEL=PRESET` | Applies a preset to a label with another name, e.g. `h2o-b512=h2o`. |
| `--revision LABEL=REV` | Records the source revision of the executable, which cannot be inferred from the checkout. |

A non-preset condition without `--expect` or `--unchecked` is refused. Without that check, a method that silently stayed off would be reported as "no effect".

How settings are applied:
- New keys that one condition adds are removed from every other condition, baseline included. A key the model already has, such as `"fsu"`, is changed only for the condition that overrides it.
- Lines matching a condition's own `--expect` patterns (or preset marker) are excluded from the fatal/truncation/fallback scan. A method may log, e.g., "truncated 4094 tokens".

Built-in presets need no `--setting` or `--expect`. A preset is used when the label is the preset name or is given `--preset`. Its `--setting` merges into the preset key, e.g. `'h2o-b512={"h2o": {"budget": 512}}'` keeps `recent=128`.

| Preset | `nntr_config.json` key | Default settings | Activation check |
|---|---|---|---|
| `snapkv` | `snapkv` | budget 256, window 16, kernel 5, maxpool | `[SnapKV cache]` line iff prompt > budget; capacity = budget + decode steps |
| `h2o` | `h2o` | budget 256, recent 128 | `[H2O] h2o_budget=... h2o_recent=...` |
| `pyramidkv` | `pyramidkv` | budget 256, window 64, kernel 5, beta 20 | `[PyramidKV]` header and per-layer retention schedule |

A baseline run fails if any preset marker or any condition's `--expect` pattern appears in its log. Every summary ends with a table of each condition's additions and activation check.

Several variants of one method can share a run and its single baseline, for example:

```bash
--condition baseline=build-main/Applications/CausalLM/nntr_causallm \
--condition capkv-r50=build-capkv/Applications/CausalLM/nntr_causallm \
--setting 'capkv-r50={"capkv": {"ratio": 0.5}}' --expect 'capkv-r50=^\[CapKV\] ' \
--condition capkv-r75=build-capkv/Applications/CausalLM/nntr_causallm \
--setting 'capkv-r75={"capkv": {"ratio": 0.75}}' --expect 'capkv-r75=^\[CapKV\] '
```

When adding a method on a new branch, print one line on activation, e.g. `std::clog << "[MyKV] budget=…"`, so `--expect` can prove it ran.

## Latency and memory (`perf`)

The protocol follows the project metric definition, with real benchmark text as input:
- one LongBench v1 case (default `gov_report:0`, an 11,040-token report to summarize) built with the official prompt and middle truncation at limits of 512 / 1,024 / 2,048 tokens. The actual prompt lengths are 508 / 1,020 / 2,044 tokens, because the chat template and the official margin are reserved. Every length must truncate the same document;
- at most 128 decode steps (`init_seq_len = prompt + 1`, as for quality); cache capacity 2,048 + 128 = 2,176, kept even when only some lengths are measured so the KV allocation stays comparable;
- original sampling and EOS, batch one, NNTR/OMP four threads, OpenBLAS one thread, Linux `taskset` `0,2,4,6`;
- one function check per condition, excluded from statistics;
- per input and condition, 3 latency trials (`--repeat`), then 1 memory trial (`--memory-repeat`) in a separate process. The condition order is rotated per trial, there is 1 s between trials and no warm-up. Sampled RSS of the original script barely varies between runs, so one memory trial suffices; latency keeps three.

```bash
python3 eval/kvcache/kvcache_eval.py perf --model models/qwen3-fp32-main \
  --official data/longbench/official-src/LongBench --data data/longbench/extracted/data \
  --condition baseline=build-main/Applications/CausalLM/nntr_causallm \
  --condition h2o=<h2o build>/Applications/CausalLM/nntr_causallm \
  --revision baseline=7d14d001 --output logs/kvcache/perf-<run-name>
```

A trial is valid only if all of the following hold:
- the prompt is echoed and the timing trailer is complete;
- the prefill token count equals the prepared prompt;
- the decode count is within the limit (early EOS is recorded);
- no fatal/truncation/fallback warning appears outside the generated text;
- the activation check passes.

A memory trial also needs the target PID, positive RSS samples, a peak equal to the largest sample, and no `smaps` read error.

`summary.md` reports mean ± sample SD, n, and the change from baseline as in the metric definition:
- time and memory: `(baseline − method) / baseline × 100`;
- TPS: `(method − baseline) / baseline × 100`.

Positive is an improvement in both. Raw values are in `trials.jsonl` and `trials.csv`. Latency comes only from latency trials and memory only from memory trials. Sampled RSS uses the script's 1 s sampling and can miss spikes. Temperature and CPU frequency are N/A on WSL.

## Output quality (`quality`)

[LongBench v1](https://github.com/THUDM/LongBench) with the protocol of `eval/pyramidkv/longbench_eval.py` on the `pyramidkv` branch (`5cdcf6f6`):
- official task prompts, middle truncation, generation limits and metrics;
- the Qwen3 chat template with thinking disabled, except for the six tasks the official script runs without chat wrapping;
- greedy decoding, one prediction per case and condition;
- `init_seq_len = prompt_tokens + 1` and `num_to_generate = max_new_tokens − 1`, because NNTrainer emits the first new token from prefill;
- SAMSum adds newline to EOS.
- Special tokens (e.g. the `<|im_end|>` text NNTrainer prints on EOS) are removed before scoring. The official prediction script decodes with `skip_special_tokens=True`; without this, an answer such as `unanswerable<|im_end|>` normalizes to `unanswerableimend` and scores 0. `records.jsonl` keeps the raw text in `raw_pred`. Scores from the pyramidkv branch's helper did not remove them and are not directly comparable.

Prompts are limited to **2,048 tokens by default**, matching the longest performance input; `max_seq_len = max_input + 513`.

Pinned inputs:

| Input | Revision |
|---|---|
| Official source (`eval.py`, `metrics.py`, `config/`) | [THUDM/LongBench](https://github.com/THUDM/LongBench) `2e00731f8d0bff23dc4325161044d0ed8af94c1e`, directory `LongBench/` |
| Dataset | [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench) `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`, `data.zip` sha256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |

Keep these local assets under the ignored `data/longbench/` directory:

```bash
mkdir -p data/longbench && cd data/longbench
git init official-src && cd official-src
git remote add origin https://github.com/THUDM/LongBench.git
git fetch --depth 1 origin 2e00731f8d0bff23dc4325161044d0ed8af94c1e && git checkout FETCH_HEAD
cd .. && python -c "from huggingface_hub import hf_hub_download; hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset', revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672', local_dir='hf-dataset')"
unzip hf-dataset/data.zip -d extracted   # v1 JSONL in extracted/data; ignore *_e.jsonl (LongBench-E)
cd ../.. && python -m pip install tokenizers jinja2 rouge==1.0.1 jieba==0.42.1 fuzzywuzzy==0.18.0
```

`fuzzywuzzy` scores `lcc` and `repobench-p`. It uses `python-Levenshtein` when
installed and `difflib` otherwise; the two can differ slightly, so the backend
is recorded in every run.

```bash
python3 eval/kvcache/kvcache_eval.py prepare --model models/qwen3-fp32-main \
  --official data/longbench/official-src/LongBench --data data/longbench/extracted/data \
  --cases data/longbench/cases/cases-2048.jsonl
python3 eval/kvcache/kvcache_eval.py quality --model models/qwen3-fp32-main \
  --official data/longbench/official-src/LongBench --cases data/longbench/cases/cases-2048.jsonl \
  --condition baseline=build-main/Applications/CausalLM/nntr_causallm \
  --samples-per-task 1 --output logs/kvcache/quality-<run-name>
```

`--samples-per-task N` and `--tasks a,b` select fixed dataset indices, independent of observed scores. The run is then labelled `FIXED_SUBSET`; do not present it as a full LongBench result. Without them all 4,750 samples of the 21 tasks run.

A prediction passes the same output checks as a performance trial. A complete run is cross-checked against the official `scorer()` per task and condition. Per-case latency is recorded, but it varies with early EOS, so use `perf` for speed claims.

`--max-input 8192` reproduces the pyramidkv branch's cases. With the Qwen3-0.6B `tokenizer.json` (sha256 `aeb13307…`), that file has sha256 `f54431f60ca437750b32efc1c2b2cf7e32a0a28eee34e7d16c01ca01b68a0495`.

## Runs and outputs

An existing `--output` directory is resumed only when the whole protocol is unchanged: inputs, selection, settings, executables, libraries, model assets, compiler and scorer. An invalid record stops a run and must be inspected first. `report DIR` rewrites `summary.md` of either kind of run.

| File | Content |
|---|---|
| `manifest.json` | protocol (kind `perf` or `quality`), hashes, environment, status |
| `trials.jsonl`, `trials.csv` | `perf`: one row per trial with raw timings, TPS, RSS, sample count, validity |
| `records.jsonl`, `scores.json` | `quality`: one record per case and condition with prediction, score and validity; per-task scores |
| `logs/…/*.{stdout,stderr}.log` | raw model output, including the `mem_usage.sh` sample log |
| `summary.md` | tables described above |

## Differences from the pyramidkv helper

- Any set of conditions and methods; settings and activation checks come from the command line, with presets for the existing branches.
- Latency/memory and quality live in one script with shared validation.
- Warning patterns are searched outside the echoed prompt and prediction.
- `--reuse-from` and the legacy Windows parser recovery are not carried over. `perf` is Linux only.
