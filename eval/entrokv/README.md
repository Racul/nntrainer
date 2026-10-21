# EntroKV prompt-cache compression

Implementation prepared on branch `entrokv` from local `main` commit
`7d14d001c9e70014de515dbd00ee6986d50d365f`.

**Status: implemented and validated on Linux/WSL2 and native Windows.
Performance evaluation completed on 2026-10-21: 36 valid measured trials per
platform, three per input/condition/phase, all with 128 decode steps.
LongBench is explicitly excluded by the updated execution instruction.**

[CPU results and interpretation](results/20261021-cpu.md),
[individual values, sample SDs and provenance](results/20261021-cpu.json),
[trial CSV](results/20261021-trials.csv).
Overall latency changes are +20.4% on Linux and -17.9% on native Windows;
sampled process peak memory changes are -0.5% and -0.9%, respectively.
KV allocation remains unchanged. One failed Linux sampler attempt was retained
separately and only that memory slot was repeated with identical artifacts.

## Algorithm contract

This ports **Entro-SnapKV**, the entropy allocator with the SnapKV eviction
operator from Appendix E.1 of
[EntroKV (Gao et al., ICML 2026)](https://proceedings.mlr.press/v306/gao26a.html).
It does not implement the paper's CaM/D2O merging variants.

For prompt length `L`, observation window `W=min(window,L)`, query head `h`:

```text
H[h] = -sum(q=L-W..L-1, j=0..q) A[q,j,h]*log(A[q,j,h]) / (W*log(L))
H[kv] = mean(H[h] for query heads sharing that KV head)
B[kv] = beta * ((1-alpha)*baseline_entropy + alpha*H[kv])
C[kv] = max(W, floor(L*B[kv]))
vote[kv,j] = mean(q in observation, h in group) A[q,j,h], j < L-W
keep[kv] = topk(maxpool(vote[kv]), C[kv]-W) + recent W positions
```

For `L=1`, entropy is defined as zero and the sole recent token is retained.
Quotas include the recent window. The minimum quota can exceed `beta*L` for
short prompts. `beta` is the Eq. 8 multiplier, **not a promise of an exact global
retention ratio**. The optional global normalization in Eq. 10 is disabled;
there is no per-layer renormalization substituting for it. Entropy is averaged
per query head before GQA aggregation, rather than computing entropy of pooled
attention. Normalization uses `log(L)` including causal zeros, per Eq. 9.

When the complete recent window covers the prompt, or beta=1/alpha=0 with a
calibrated prior of 1, all tokens are known to be retained. Those cases preserve
the original attention path exactly and skip entropy reduction.

Otherwise the final observation window is collected from ordinary attention probabilities
as attention outputs are computed in blocks of at most 32 queries. There is no
additional QK matmul and no new full-prompt squared probability plane. Max-pool
ignores positions outside the prefix (equivalent to negative-infinity padding).
Ties prefer smaller original positions. Kept positions are stored chronologically
with the same permutation for K and V. Eviction runs only after all prefill
outputs have consumed the original values. Generated tokens are appended without
further eviction, following the one-prefill SnapKV schedule.

The current attention integration supports CPU, NCHW, dense **Qwen3**, batch 1,
FP32 activations/scores, FP16/UINT16 KV storage (and FP32 Tensor policy checks),
one complete prefill, and consecutive one-token decode. Reported measurements
use FP32 weights; other weight precisions remain unevaluated.
GPU, FP16 activations, internal cache, chunked/resumed prefill, multi-turn reuse,
cache save/load, sliding windows, ring caches, attention sinks, logit softcapping,
skip-prefill and int8 KV are rejected on the enabled path. Missing settings or
`beta=0` retain the existing path and do not impose these restrictions.

## Position and memory contract

Each KV head occupies its own compact prefix in the existing token-major slab.
Rows between its quota and the largest head quota are zero-filled and masked
with negative infinity before decode softmax. Shared generated-token rows start
after the largest quota. Thus physical cache writes use
`max(C) + logical_position - L`; Q/K RoPE and model/session counters keep the
original logical positions. Already-rotated keys are never rotated again.

**KV allocation remains unchanged.** Attention reads `max(C)+generated` rows,
including padding, rather than a ragged sum of head quotas. This implementation
does not release the cache manager's maximum-capacity buffers. Reports must
distinguish the sum of live head quotas, padded physical span, allocation bytes,
and process peak memory. The enabled block attention path differs from the main
flash/GEMM prefill path; performance and memory effects require measurement.

## Configuration

Add this field to a temporary copy of the existing `nntr_config.json`:

```json
{
  "entrokv": {
    "beta": 0.3,
    "alpha": 1.0,
    "window": 32,
    "kernel": 7
  }
}
```

`alpha=1` is an explicit instance-only setting, requiring no invented model
calibration. For `alpha<1`, supply a model-specific `baseline_entropy` in `[0,1]`
from a separate held-out calibration workload. Do not reuse the paper's Qwen3-8B
constant for Qwen3-0.6B or calibrate on the 21 evaluation samples. Window/kernel
above follow a paper configuration; they are not a validated Qwen3-0.6B setting.
Use `NNTR_ENGINE=cpu` on builds with device backends. Remove other KV-method
configuration blocks. Unknown settings, nonintegral window/kernel, out-of-range
or nonfinite values are rejected. A completed prefill emits one
`[EntroKV] active ... quotas=... allocation=unchanged` marker per MHA layer;
configuration alone is not evidence of activation.

## Validation coverage

- `unittest_entrokv`: real FP32/UINT16 Tensor storage, entropy/quota differences,
  matching K/V row selection, deterministic ties, recent tokens, padding masks,
  full-budget/single-token boundaries, consecutive decode bookkeeping, invalid
  inputs, and fresh-state requirements.
- `unittest_causallm_models`, filter `Qwen3EntroKV.*`: nonzero deterministic tiny
  Qwen3 projections, full-budget logits and multiple decode steps versus baseline,
  actual `run()` activation, cache resume rejection, and disabled-run equivalence.
- Existing `unittest_kv_cache_manager` and model tests remain required regressions.

Source inspection, clang-format **14.0.6**, and `git diff --check` were performed.
Execution outcomes and platform limits are recorded in the result report.

### Completed build and function checks

Measured implementation: `31f92a73cd7c0690158dedb29cbef8d5b7877512`.

| Platform | Check | Outcome |
| --- | --- | --- |
| Linux/WSL2, GCC 13.3 | EntroKV Tensor policy | 15 passed |
| Linux/WSL2, GCC 13.3 | Existing KV cache manager | 20 passed |
| Linux/WSL2, GCC 13.3 | CausalLM model suite | 83 passed, 1 ARM-only QINT4 case skipped |
| Native Windows, MinGW GCC 13.2 | Same Tensor policy source | 15 passed |
| Both | Actual tiny CLI: absent/zero/full-retention/compressed | All four passed; each prefill 16, decode 8 |
| Both | Full-retention/disabled/zero output comparison | Identical generated output |
| Both | Performance helper preflight | Passed; includes build options, assets and sampler API |

The model tests include nonzero Q/K/V projections, compressed prefill logits
versus the original path, multiple decode steps, actual `run()` activation,
disabled equivalence, unsupported cache resume and incompatible GQA rejection.
Negative cases meet the repository's negative/positive test-count convention.
The native policy runner compiles the existing test source with native compiler
flags and links the verified core API; it does not substitute WSL tests for
native execution. The broader model suite was run on Linux, with actual CLI
function checks on both systems.

An initial premeasurement function check found full-retention output differences
when routed through the alternative attention arithmetic. Guaranteed-full-cache
cases now use the original attention path, and the corrected build passed the
same check. Failed validation attempts are retained in the validation directory;
none contribute to performance statistics.

## Authorized performance evaluation (LongBench excluded)

Use the independently built local main baseline above and a separate EntroKV
build, with source/header/compiler/flag verification before reusing unchanged
core libraries. All CausalLM feature objects build fresh. Existing validated
Qwen3-0.6B assets are reused; weights, tokenizer, configuration, runtime libraries
and binaries are identified by checksums.

Protocol: FP32 weights/activations, FP16-bit UINT16 KV, batch 1, inputs
512/2048/4096, capacity **4225**, at most **128** decode steps, beta **0.3**, alpha
**1**, window **32**, max-pooling kernel **7**. No calibration prior or Eq.10
normalization. NNTR/OMP threads **4**, OpenBLAS **1**. Linux affinity uses allowed
CPUs 0,2,4,6; native Windows affinity is unpinned. Preserve original sampling,
EOS and default sampling seed 12, and record actual generated counts. Each
phase/input/condition uses three valid fresh-process trials, with condition order alternated by trial;
function checks are excluded. There are **36 measured trials per platform**.

Latency and memory are separate processes. Main latency is CLI `total:` excluding
model load; e2e and stage times are supplemental. Linux uses the unchanged
`Applications/utils/mem_usage.sh` model-PID smaps RSS sampler (~1 second). Native
Windows uses current WorkingSetSize via PSAPI at 50 ms. Both are sampled maxima,
with OS peaks stored separately. No synthetic trials, thermal controls or quality
scores. Failed attempts remain in the validation directory.

`entrokv_eval.py --help` documents arguments; `--check` verifies assets and build
settings without inference. Example (substitute build and model paths):

```bash
python3 eval/entrokv/entrokv_eval.py /path/to/model --check \
  --baseline-bin /path/to/main/nntr_causallm \
  --entrokv-bin /path/to/method/nntr_causallm
python3 eval/entrokv/entrokv_eval.py /path/to/model \
  --baseline-bin /path/to/main/nntr_causallm \
  --entrokv-bin /path/to/method/nntr_causallm --repeat 3 --output /path/to/new-run
```

The native Windows run uses native Windows Python and `.exe` paths with the same
helper. Evaluation-only MinGW compatibility flags/exports and the prior verified
main-return exit adapter are applied symmetrically to baseline and method builds,
without editing repository or vendored sources. The adapter runs after model
and local cleanup, preserves main's exit code, flushes output and bypasses DLL
global shutdown; it does not turn timed-out inference into success.

After all valid trials, `report.py` audits counts and emits platform-specific
means, sample SDs, actual decode counts and mean-change tables. **LongBench is
not part of this evaluation.**

## Source provenance

The implementation follows the primary paper, Eq. 4/8/9 and Appendix E.1,
including its explicit GQA mean-entropy guidance. The author's GitHub software
link returned HTTP 404; the anonymous repository API returned HTTP 410 during
this task. Author-code parity could therefore not be checked and is not claimed.
Downloaded official PMLR PDF SHA-256:
`a255eee643e732470f46379fc729fb12785b57dbfd11dfaefd3601ecba1e208a`.
