# H2O CPU evaluation

The policy lives in `Applications/CausalLM/layers/h2o.h`. The only existing
runtime files changed are `mha_core.{h,cpp}` and `causal_lm.{h,cpp}`; model
graphs, cache manager, kernels, Meson files and existing tests are unchanged.

Add this field to the model's existing `nntr_config.json`:

```json
"h2o": { "budget": 256, "recent": 128 }
```

`budget` includes the recent-token quota: this example keeps 128 heavy hitters
and the 128 most recent tokens per KV head. Omit `h2o` or use `budget=0` to
disable it. Active settings require `0 < recent < budget`.

The full causal prefill seeds scores from all query rows. Each decode adds
softmax attention to the surviving tokens' accumulated scores, appends the
new token, and selects heavy hitters outside the recent quota. K, V and scores
are compacted together after the attention output is computed. Up to `budget`
tokens remain after each step; decode temporarily attends to `budget + 1`.
This follows the official [H2O real-drop implementation](https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_real_drop/modify_llama.py)
of the [H2O paper](https://arxiv.org/abs/2306.14048).

This is a **shared KV-head GQA variant**: query-head scores within a KV group
are averaged, matching the cache layout used by the SnapKV port. It does not
reproduce independent query-head selection. Absolute RoPE positions remain
unchanged. Support is limited to Qwen3, CPU, batch one, a fresh full prefill,
then consecutive single-token decode. Cached/skipped/chunked prefill, resumed
sessions, training, attention sinks, sliding windows, ring caches, int8 KV,
and GPU/CUDA engines are rejected when H2O is active.

The existing KV allocation and tensor shapes remain unchanged, as specified
in `refs/snapkv_plan.md`. This bounds the KV span read during decode; it does
not implement allocation shrinkage. Explicit prefill probabilities are needed
for scoring, so prefill time/peak RSS can increase relative to fused attention.
Performance and answer quality must be measured; neither is assumed improved.

## Entry points

All helpers use existing executables and model assets. They do not build or
deploy anything. Python needs `tokenizers`. Each condition uses exact repeated
token-5555 prompts, lengths 512/2048/4096, at most 128 decode steps, capacity
4224, batch one, NNTR/OMP threads four and OpenBLAS threads one by default.
Latency and memory use separate processes, five repetitions per phase and
condition, with alternating baseline/H2O order. Initial function checks are
excluded from statistics. Original sampling/EOS settings are preserved.

Linux and Windows share `h2o_eval.py`; Android uses the same result parser and
summary format. `--help` lists protocol overrides. An H2O activation marker
is required in optimized runs, preventing old binaries from silently being
reported as H2O. `--check` only validates prerequisites; function checks prove
runtime activation when the full evaluation starts.

Use `--condition baseline` to measure only the baseline. This requires only
`--baseline-bin`, disables H2O and SnapKV in temporary model settings, and
runs one excluded function check plus the requested latency/memory trials.
`--condition h2o` measures only H2O; the default `both` compares both modes.

### Linux

```bash
./eval/h2o/h2o_eval.sh /path/to/qwen3-fp32 --check
./eval/h2o/h2o_eval.sh /path/to/qwen3-fp32 --condition baseline \
  --baseline-bin /path/to/main/build/Applications/CausalLM/nntr_causallm
./eval/h2o/h2o_eval.sh /path/to/qwen3-fp32 \
  --baseline-bin /path/to/main/build/Applications/CausalLM/nntr_causallm \
  --h2o-bin ./build/Applications/CausalLM/nntr_causallm
```

Default builds are `build-baseline/` and `build/`. `BASELINE_BIN`, `H2O_BIN`,
`PYTHON` and `EVAL_CPUS` can override defaults; `--cpus` defaults to `0,2,4,6`.
Build both with matching Meson options. For a same-binary on/off comparison,
point both binary options to the H2O build and record that distinction.
Memory uses the unchanged `Applications/utils/mem_usage.sh` on the actual
model PID. `--resume OUTPUT_DIR` resumes matching valid completed trials.

### Android

Deploy the H2O CPU build separately using the existing Android build/install
instructions. Both conditions use this same executable: baseline omits the
H2O setting, optimized runs enable it.

```bash
./eval/h2o/android_h2o_eval.sh --check --adb /path/to/adb
./eval/h2o/android_h2o_eval.sh --adb /path/to/adb
```

The existing device model defaults to
`/data/local/tmp/nntrainer/causallm/models/qwen3-fp32`. Use `--model`, `--serial`,
`--condition baseline|h2o|both`, `--phase latency|memory|both` or
`--expected-binary-sha256` as needed. Windows ADB in WSL is supported, as in
the SnapKV helper. Latency uses the existing `run_causallm.sh`; memory invokes
the model binary directly through the original `mem_usage.sh`.

The device config is backed up before modification and restored with a hash
check on success, failure or interruption. Stale backup/temporary config files
stop the run. Raw invalid trials are retained and stop evaluation. Cooling
uses the existing thermal-zone policy between contexts after the first,
stopping on read failure or timeout; it does not cool before every trial.
After interruption, inspect device processes before retrying.

### Windows

Use native Windows Python and matching CPU Meson builds, as in the SnapKV
Windows helper. Default builds are `build-causallm-win-baseline/` and
`build-causallm-win/`. DLL directories are discovered per build.

```powershell
.\eval\h2o\windows_h2o_eval.ps1 C:\models\qwen3-fp32 --check
.\eval\h2o\windows_h2o_eval.ps1 C:\models\qwen3-fp32 --condition baseline --baseline-bin C:\baseline\Applications\CausalLM\nntr_causallm.exe
.\eval\h2o\windows_h2o_eval.ps1 C:\models\qwen3-fp32 --baseline-bin C:\baseline\Applications\CausalLM\nntr_causallm.exe --h2o-bin C:\h2o\Applications\CausalLM\nntr_causallm.exe
```

Windows memory is sampled Working Set from PSAPI, in KiB. It differs from
Linux/Android smaps RSS and is reported under a separate metric name. CPU
pinning and thermal control are not performed on Windows.
If the current execution policy blocks `.ps1` files, use a process-scoped
invocation: `powershell -NoProfile -ExecutionPolicy Bypass -File .\eval\h2o\windows_h2o_eval.ps1 ...`.

## Results and limits

Results default to `logs/h2o/<platform>/<timestamp>/`: raw logs, configs,
`<platform>-comparison.json`, `<platform>-comparison.csv` and `summary.md`.
Desktop model configs use temporary copies. Asset/binary/runtime hashes,
build options where available, settings and actual token counts are recorded.
Build provenance must be verified separately; checkout HEAD does not prove
the revision of an existing executable. Use `--output` to select a directory.

The primary memory metric is sampled RSS on Linux/Android and sampled Working
Set on Windows. OS peaks stay separate. Samples can miss brief peaks; the
original RSS script recommends runs of at least 30 seconds. Invalid/empty
RSS is never replaced by an OS peak. Means, sample SD and valid counts are
reported per mode and phase. Check actual decode counts and early EOS before
claiming an equal-work speedup. There is no answer-quality scoring.

## Implementation validation

Linux x86 CPU validation used GCC, `enable-fp16=false`, FP32 model tensors
and the existing UINT16 cache containing FP16 bits:

- The CausalLM CLI, quantizer and existing cache/model test targets built.
- Cache tests: 20 passed. Model tests: 79 passed, one existing ARM-only QINT4
  case skipped on x86.
- A temporary real-Tensor oracle compared chronological heavy-hitter/recent
  selection and score migration over 14 FP32/UINT16 scenarios, two sessions
  and 21 steps per scenario. Config, sequence and probability guards passed.
- A deterministic tiny two-layer Qwen3 ran through the actual CLI. Disabled,
  budget-zero and uncompressed outputs matched; budget 8/recent 4 completed
  full prefill and eight decode steps. Invalid quotas and cached/chunked
  prefill were rejected.
- The Linux helper completed function, latency and memory phases using that
  same executable with H2O off/on, prompt lengths 16/32 and one repetition.
  All ten rows were valid; original RSS logs, JSON, CSV and summary were saved.
- Native Windows verified PowerShell parsing/launcher help and PSAPI sampling
  of the parent and a child process. Android controller simulation verified
  alternating modes, config restoration on success/failure and invalid-row
  retention. Desktop simulation also covered interrupted-run resume.

Validation artifacts are ignored under `build/validation/`. These checks do
not measure production-model performance or answer quality. Android inference,
Android FP16 attention were not executed.

Native Windows production-model evaluation subsequently completed using
Qwen3-0.6B with FP32 weights/activations and existing 16-bit KV storage.
Contexts were 512/2048/4096, capacity 4224, batch 1, NNTR/OMP 4 and OpenBLAS 1;
H2O used budget 256/recent 128. Baseline used three measured trials per phase
and context; H2O used one. Both builds used the same evaluation-only main-return
adapter to avoid a MinGW DLL shutdown deadlock. The 4096-token case reports
127 decode steps due to the existing capacity boundary; other cases report 128.
Results and build provenance are saved under `build/validation/`, with the
comparison in `windows-h2o-comparison/summary.md`. Synthetic augmentation uses
the baseline sample SD symmetrically around the single H2O measurement, so its
mean equals that measurement; it adds no measured trials. Peak memory is sampled
Windows Working Set. H2O retains fewer cache entries but preserves allocated KV
tensor sizes and adds attention-scoring buffers. Answer quality was not measured.
