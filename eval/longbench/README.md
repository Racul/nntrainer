# LongBench v1 quality evaluation

`longbench_eval.py` measures output quality of a baseline CausalLM executable
and KV-cache method executables (`snapkv`, `h2o`, `pyramidkv`) on
[LongBench v1](https://github.com/THUDM/LongBench). It follows the protocol of
`eval/pyramidkv/longbench_eval.py` on the `pyramidkv` branch (`5cdcf6f6`), so
prepared cases and scores are comparable with that branch's results, but it
accepts any number of conditions and is not tied to one method.

The script does not build anything. Build each condition from its own branch
with identical Meson options, then pass the executables. All conditions of one
run share one model directory, so compare executables built on the same base.

Convert the model with the converter of that base. Since `5103c857` the Qwen3
converter writes FFN weights as gate, up, down, and main no longer remaps the
old up, gate order (`1829e4de`). A `.bin` made by an older converter loads on
main without error but generates fluent-looking garbage; the validity checks
below cannot detect that, so inspect a few predictions of every new setup.

## Pinned sources

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

## Usage

Prepare the cases once per context limit. The prompts use the official task
templates, official middle truncation and the Qwen3 chat template with
thinking disabled (six official tasks are run without chat wrapping).

```bash
python3 eval/longbench/longbench_eval.py prepare --model models/qwen3-fp32 \
  --official data/longbench/official-src/LongBench --data data/longbench/extracted/data \
  --cases data/longbench/cases/cases-8192.jsonl --max-input 8192
```

With the pinned sources and the Qwen3-0.6B `tokenizer.json` (sha256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`), the
8192-token cases file has sha256
`f54431f60ca437750b32efc1c2b2cf7e32a0a28eee34e7d16c01ca01b68a0495`, the same
file the pyramidkv branch evaluated.

Validate inputs with `--check`, then run. Each `--condition` is `NAME=BINARY`.
Method settings default to the values below and can be overridden with
`--config NAME=JSON`. `--revision NAME=REV` records the source revision of
each executable, which cannot be inferred from the checkout.

```bash
python3 eval/longbench/longbench_eval.py run --model models/qwen3-fp32 \
  --official data/longbench/official-src/LongBench --cases data/longbench/cases/cases-8192.jsonl \
  --condition baseline=build-main/Applications/CausalLM/nntr_causallm \
  --condition snapkv=build-snapkv/Applications/CausalLM/nntr_causallm \
  --config snapkv='{"budget": 256}' --revision baseline=7d14d001 --revision snapkv=<sha> \
  --samples-per-task 1 --output logs/longbench/<run-name>
python3 eval/longbench/longbench_eval.py report logs/longbench/<run-name>
```

| Method | `nntr_config.json` key | Default settings | Activation check |
|---|---|---|---|
| `snapkv` | `snapkv` | budget 256, window 16, kernel 5, maxpool | `[SnapKV cache]` line iff prompt > budget; capacity = budget + decode steps |
| `h2o` | `h2o` | budget 256, recent 128 | `[H2O] h2o_budget=... h2o_recent=...` |
| `pyramidkv` | `pyramidkv` | budget 256, window 64, kernel 5, beta 20 | `[PyramidKV]` header and per-layer retention schedule |

A baseline run fails if any method marker appears.

`--samples-per-task N` and `--tasks a,b` select fixed dataset indices,
independent of observed scores. The run is then labelled `FIXED_SUBSET`; do not
present it as a full LongBench result. Without them all 4,750 samples of the
21 tasks run.

An existing `--output` directory is resumed only when the whole protocol
(cases, selection, settings, executables, libraries, model assets, scorer) is
unchanged. Invalid predictions stop the run and must be inspected first.

## Protocol

- Greedy decoding (`do_sample=false`, `top_k=1`, seed 42), one prediction per
  case and condition, CPU engine, batch one, NNTR/OMP four threads, OpenBLAS one
  thread, Linux `taskset` affinity `0,2,4,6` by default.
- NNTrainer emits the first new token from prefill, so a case runs with
  `init_seq_len = prompt_tokens + 1` and `num_to_generate = max_new_tokens - 1`
  to produce the official per-task generation length.
  `max_seq_len = max_input + 513`.
- SAMSum adds newline to EOS, as in the official prediction script.
- A prediction is valid only if the exact prompt is echoed, the timing trailer
  is complete, the prefill token count equals the prepared case, the decode
  count is within the limit, no fatal/truncation/fallback warning appears in
  the log outside the generated text, and the activation check passes.
- Per-case scores use the official metric (first line only for `trec`,
  `triviaqa`, `samsum`, `lsht`). A complete run is cross-checked against the
  official `scorer()` per task and condition.
- Per-case latency is recorded but varies with early EOS. Use the separate
  latency/memory protocol for performance claims.

## Outputs

| File | Content |
|---|---|
| `manifest.json` | protocol, hashes, status |
| `records.jsonl` | one record per case and condition: prediction, score, validity, timings |
| `logs/<task>/<index>-<condition>.{stdout,stderr}.log` | raw model output |
| `summary.md`, `scores.json` | per-task scores, differences from baseline, mean of complete tasks |

## Differences from the pyramidkv helper

- Any set of conditions; method settings and activation checks come from one
  table instead of PyramidKV-only arguments.
- Warning patterns are searched outside the echoed prompt and prediction, so a
  prediction containing a word such as "truncated" is not rejected.
- The run does not import the latency helper, so it has no build-option
  requirements beyond identical options across conditions and a CPU build.
- `--reuse-from` and the legacy Windows parser recovery are not carried over.
