# OBCache CPU implementation

Branch `obcache` starts at local main
`7d14d001c9e70014de515dbd00ee6986d50d365f`.
On 2026-10-21 the user authorized builds, tests, Linux/native Windows
performance measurements and a report/PR. LongBench is explicitly excluded
from this run. Validation and measurement results are linked below.

## Algorithm and configuration

The policy follows [OBCache](https://arxiv.org/abs/2510.07651), equations
(5)-(7), and the official
[`OBCacheSnapKVPress`](https://github.com/DreamSoul-AI/OBCache/blob/189ffadd7b2f29eaf311e67f39dbf2a26f5f4dd3/kvpress/presses/obcache_press.py)
at revision `189ffadd7b2f29eaf311e67f39dbf2a26f5f4dd3`.
OBCache provides scores for an eviction policy; this port uses the official
SnapKV observation-window policy with a uniform absolute prompt budget.
It does not use PyramidKV layer budgets or H2O online eviction.

Add this optional object to a copy of `nntr_config.json`:

```json
"obcache": {"budget": 256, "window": 64, "kernel": 5, "score": "key"}
```

Omitting the object or setting budget to zero preserves the existing inference
path. Budget includes the protected recent window. Window must be positive
and smaller than budget, and kernel must be a positive odd integer.
The default score is OBCache-K (`key`); `value` and `joint` expose the paper's
other two scores. No attention-only substitute is used.

For observation queries in the final window, let `Z = Q K^T / sqrt(d)` and
`A = softmax(Z)` with the full causal key span. The port matches the official
`gqa_aggregation=pre_redc`: average A and Z over query heads sharing a KV head
before evaluating the score, then recompute `O = mean(A) V`. O includes
protected values. Scores for each older token are:

| Setting | Score summed over observation queries |
| --- | --- |
| `value` | `A² * ||V||²` |
| `key` | `(A*Z)² * ||V-O||²` |
| `joint` | `A² * ||V + Z*(V-O)||²` |

The joint expression is algebraically equivalent to equation (7) including
its cross term, while avoiding subtraction of large squared norms.
Scaled logits are preserved before softmax; logarithms of probabilities are
not a substitute for these logits. Scoring uses the actual cached V, including
half-storage rounding. Scalar reductions use double precision, so rankings
near numerical ties can differ from the official FP16/BF16/FP32 computation.

Scores receive zero-padded average pooling of width kernel. Each KV head
retains its highest `budget-window` prefix scores and every window token.
Equal-score ties prefer earlier positions. Selection happens after the original
prefill output is complete; all scoring finishes before any K/V is moved.
Selected K/V move together in chronological order. This deterministic tie
rule and chronological storage differ from upstream top-k ordering. Explicitly
protecting every observation token also resolves upstream's potential ties
between the padded maximum window score and the highest prefix score.

Subsequent decode tokens append to the compact cache without additional
eviction. Query/key RoPE uses the original absolute token position; attention
uses the compact physical span. There is no RoPE rerotation of retained keys.
Prompts of length at most budget skip scoring and compaction entirely.

The implementation retains existing full-capacity KV allocations. It reduces
the effective decode attention span, not allocated KV bytes. Scoring keeps
one query row of logits/probabilities at a time and prefix scores per KV head;
it does not add a full triangular prefill score tensor. Scoring temporaries
can still increase process peak memory. Measured latency and memory results are linked below.

## Supported path and validation

The enabled path is restricted to Qwen3 FP32 weights/activations, CPU, batch
one, one fresh full causal prefill and consecutive single-token decode.
FP32, UINT16 (FP16 bits) and FP16 KV storage are accepted with matching K/V
shape and NCHW layout. The normal no-FP16 build stores KV as UINT16 bits;
FP32 activations do not imply FP32 KV allocation.

Enabled-path guards reject training, device engines, FP16 activations,
int8 KV, sliding windows, sinks, ring caches, nonzero logit softcapping,
skipped/chunked/cached prefill, continuation and serialized KV caches.
Android builds that narrow FP32 activations into the FP16 attention path are
also rejected. Disabled inference retains its existing support.

`test/unittest/models/unittest_obcache.cpp` is registered in the existing
`unittest_causallm_models` target. Tests have been written for the three
scores, pre-score GQA reduction, protected values in O, independent head
selection, K/V correspondence in real FP32/UINT16 tensors, pooling and ties,
budget boundaries, session reset, multiple decode steps, invalid inputs and
tiny deterministic Qwen3 off/on activation. Linux OBCache tests (12: six positive and six negative), existing Qwen3 tests (6), and KVCacheManager tests
(20) passed. Tiny CLI off/zero/uncompressed/compressed checks passed on both
Linux and native Windows. The Windows test build uses the existing MinGW
main-return adapter after model/local cleanup; DLL global shutdown is excluded.

Build independent main/OBCache trees and run the
new tests plus relevant existing Qwen3 and KVCacheManager tests. Match compiler,
Meson options, runtime libraries, precision and threads within each platform;
record source snapshots (including new files) and binary hashes.

## Performance helper

Run `python3 eval/obcache/obcache_eval.py --help` and `--check` first. The same
entrypoint supports Linux Python and native Windows Python/PSAPI. The helper
uses temporary model configurations, preserves the original sampling/EOS
settings, and checks every layer's retention and score/GQA markers.

```bash
python3 eval/obcache/obcache_eval.py models/qwen3-fp32 \
  --baseline-bin /path/to/main/build/Applications/CausalLM/nntr_causallm \
  --obcache-bin /path/to/obcache/build-runtime/Applications/CausalLM/nntr_causallm \
  --repeat 3 --cpus 0,2,4,6 --output /path/to/new-results
```

On Windows, use the native Windows interpreter and `.exe` paths. Use
`--resume OUTPUT` only with a matching saved protocol and unchanged assets,
binaries, runtime libraries and helper. Invalid trials remain recorded and stop
the controller. Source/build provenance must be recorded separately from the
checkout HEAD. `report.py RESULTS_ROOT` audits both completed platform runs and
renders the tables. LongBench is outside this helper's scope.

## Authorized evaluation and PR

The current scope excludes LongBench:

| Work | Required scope after signal |
| --- | --- |
| Performance | Linux and native Windows, baseline and OBCache, three trials each per input and phase |
| Inputs | Existing protocol to verify: 512/2048/4096 input tokens, 128 decode steps, capacity 4225, batch one, NNTR/OMP 4, OpenBLAS 1 |
| Phases | Separate latency and memory processes; function checks excluded from statistics |
| Memory | Linux original mem_usage.sh sampled model-PID smaps RSS; Windows sampled native process Working Set |
| Quality | LongBench excluded by the user; no quality scores in this run |
| Reports | Tables for each platform, every trial, mean change, activation and provenance records; quality remains unmeasured |
| PR | After evaluation, use the repository's exact PR template, DCO sign-off and agent co-author trailer |

For the three-input performance protocol, this requires 36 measured processes
per platform: 2 conditions × 3 inputs × 2 phases × 3 trials. No LongBench
predictions are authorized in the current scope.
Reuse the established evaluation helpers from the other method branches,
adapting configuration and activation validation to OBCache. Validate markers
for every layer, including retained length, score and `gqa=pre_redc`.
Keep raw logs and invalid trials. Do not report quality scores in this run.

## Completed performance evaluation

Linux and native Windows each completed 36 measured trials: three baseline
and three OBCache runs per input and phase. Every measured run decoded 128
steps. LongBench was excluded by the user. Four extra negative unit tests were
added after collection to satisfy the contributing guide; inference sources and
measured binaries did not change. The original evaluated source files and the
pre-expansion unit-test logs remain archived with their hashes.

The legacy Linux helper's auxiliary OS-peak field read the CLI's honest
RssAnon+accelerator footprint. The published helper now reads OS ru_maxrss from
`Max Resident Set Size`, and the report reparses that original log line while
preserving the legacy CLI value and raw manifests. Primary latency and sampled
RSS results are unchanged. The exact helper used during collection is archived
at `build/validation/obcache-run/evaluation-helper-used.py` for provenance/resume.

Results: [CPU performance tables](results/20261021-cpu.md) and [audited trials/provenance](results/20261021-cpu.json).
