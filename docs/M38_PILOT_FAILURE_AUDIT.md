# M38 - Aggregate-only failure-analysis baseline and research protocol

## Scope (post-hoc, NOT a detector improvement)

This stage closes the *exploratory* M37 3-clip validation diagnostic
loop and prepares clean experimental comparisons. The input reports were
produced on the owner's Ubuntu machine from an author-hosted
RepCount-A `LLSP` TAR.GZ, with actual source annotation hashes and
no online rehosting. **GitHub CI only tests synthetic report fixtures.**
These are not independently replicated cross-person accuracy estimates.

Do not edit original M34 thresholds or train on the three validation
videos and then claim unbiased validation on those same clips.

## Observed pilot (owner-supplied, 27 October 2026)

| Evidence | 3-video pilot |
| --- | ---: |
| Source annotated Squat repetitions | 45 |
| M34 predicted repetitions | 1 |
| Event TP / FP / FN | 1 / 0 / 44 |
| Event precision (just one prediction) | 1.0 |
| Event recall | 0.02222 |
| Event F1 | 0.04348 |
| M34 mean absolute count error per video | 14.6667 |
| Always-zero predicted count MAE per video | 15.0 |
| M34 MAE absolute advantage | 0.3333 reps/video |
| Accepted counter observations | 210 / 210 |
| Ticks with measurable 2D knee geometry | 185 / 210 |
| State-phase samples `await_return` | 4 / 210 |
| Broken consecutive low confirmations | 18 |
| Publisher windows with <6 valid angle observations | 44 / 45 |
| Publisher windows containing >=2 high AND >=2 low ticks | 3 / 45 |
| Publisher windows with a low-then-high observation | 16 / 45 |
| Source videos excluded | 0 |

The ever-zero baseline predicts 0 counts for each of the 3 source
videos. All counts in source annotations are nonnegative, so its
per-video MAE is total annotated reps divided by number of videos:
`45 / 3 = 15`. It never produces a positive event, meaning
**event precision and event F1 are undefined (null)** with the
current evaluation convention; event recall is 0 when source
positives exist. This comparison establishes only an absolute
MAE difference of ~0.33 reps per video, not statistical
significance or reliable superiority.

Observed 44/45 publisher intervals have fewer than six *usable*
measurements; this does **not** mean 44 actual reps are inherently
uncountable. An annotation interval may omit a preliminary high
state or the return endpoint needed by M34. The inference that
temporal sparsity *causes* the undercount would require controlled
independent experiments. Phase duration resets and side switches
may contribute, but are not established per-rep causes.

## One-command consistency and baseline audit (no 8.9 GB reread)

From repo root on Ubuntu, use the three reports already saved in
`data/processed/`:

```bash
git pull --ff-only
source .venv-m20/bin/activate

python -m scripts.m38_pilot_aggregate_audit \
  --pilot data/processed/m37_repcount_squat_event_results.json \
  --observability data/processed/m37_repcount_squat_observability_pilot.json \
  --phase data/processed/m37_repcount_cycle_audit.json \
  --report data/processed/m38_pilot_aggregate_baseline_audit.json
```

The script verifies identical source annotation SHA256, split,
sample size, event confusion counts and reported metrics across
runs; validates M37.1 and M37.2 counter totals; computes an
always-zero count-error baseline; writes **only aggregate values**
into a Git-ignored file. It refuses mismatched sources, bad
counts or overwriting an old report. No raw source videos,
poses, per-cycle timestamps, or IDs are read, saved or pushed.

## M38 development/evaluation preregistration before testing any new model

**Research question:** Can a temporally appropriate repetition
detector improve event-level recall and session count MAE on
independently annotated Squat video while controlling
false-positive rep counts and tracking abstentions?

**Dataset boundaries:** The previously viewed 3 validation
clips are **exploratory diagnostic cases**. They cannot be
treated as untouched held-out outcomes for model selection.
New candidates (signal-processing, temporal-state or
learning-based) must be designed and tuned on an explicitly
declared publisher training/development cohort. Keep the
publisher TEST split completely unseen until the candidate
family, hyperparameters, event-matching tolerance, source
preprocessing, seed and selection rule are frozen and logged.
If matching subjects/scenes cross publisher splits, document
that leakage risk and report any achievable subject-disjoint
evaluation separately. Do not claim subject-disjointness from
filename alone. Check authorization and dataset provenance.

**Frozen comparison protocol:** On the same allowed annotated
clips with a matched pose extractor and time grid, compare
(1) always-zero count control, (2) original M34, (3) an
untuned descriptive angle-extrema baseline, and (4) a
new temporal model trained solely on development data (when
available). Keep exact event end matching one-to-one, with
the original `min(2500 ms, max(750 ms, 0.35*GT rep duration))`
tolerance fixed for headline M37-compatible results.
Log alternate predeclared tolerance sensitivity analyses
only *in addition*, never optimize tolerance after observing
TEST outcomes.

**Required reporting:** number of videos, subjects if securely
available and consent/terms permit; annotated positive events;
source zero-rep negative controls; video-level MAE, bias, exact
count accuracy, within-one fraction; event TP/FP/FN and
precision/recall/F1 with null denominators handled; tracking
coverage and exclusions; computational cost and latency where
measurable. Report macro video-level summaries as well as
micro event totals, with transparent denominators. For
uncertainty, bootstrap at **participant level** only when
reliable participant grouping is available and large enough;
otherwise disclose that 3 clips cannot support meaningful
population confidence intervals or hypothesis tests.

**Ablations only after clean cohort separation:** pose sampling
cadence; handling of missing pose; side switching; phase
hysteresis/temporal memory; view angle; Python-vs-browser
estimator parity. Perform selected ablations on development
data only. Do not claim the source annotation bounds are
frame-perfect without direct provenance/synchronization checks.

**Release gates:** Training and source metrics are separate;
do not imply physical form correctness or injury-prevention
accuracy from repetition detection. Keep personal videos,
raw frames, landmarks and source data under ignored local
paths, never under git. Genuine independent paper-quality
claims require more than this three-video negative pilot.

## Current conclusion

The verified research result so far is a *negative finding*:
the original on-device M34 conservative counter detects 1/45
annotated Squat events across 3 exploratory RepCount validation
videos, while raw knee-angle observations were often available.
The next scientifically justified work is to **build and
evaluate a separate temporal baseline using development-only
annotations**, then test its generalization on untouched
publisher test data if authorized. No new detector is claimed
by this M38 audit.
