# M32 - Bicep Curl negative result, frozen reference, and research card

## Owner-provided M31 findings

M31 reproduces **every M30 metric exactly** with 15 original TRAIN
participants and 90 video-level Bicep Curl source clips. Source M20
validation and TEST are **not scored**.

| Source model | Group5 BA | LOSO BA | LOSO publisher bad recall |
| --- | ---: | ---: | ---: |
| baseline_264 | 60.00% | **60.00%** | **62.22%** |
| kinematic_276 | 57.78% | 56.67% | 55.56% |
| tracking_proxy_268 | 61.11% | 58.89% | 60.00% |
| temporal_280 | **62.22%** | 57.78% | 55.56% |

The paired **LOSO** descriptive participant-cluster bootstrap
intervals for BALANCED ACCURACY changes versus baseline are:

| Model | Delta BA | 95% interval | Proportion of positive bootstrap deltas |
| --- | ---: | ---: | ---: |
| kinematic_276 | -3.33 points | [-7.78, +1.11] points | 6.3% |
| tracking_proxy_268 | -1.11 points | [-4.44, +2.22] points | 17.6% |
| temporal_280 | -2.22 points | [-7.81, +2.22] points | 13.5% |

**Every interval overlaps zero**, and 1,000 clustered resamples
of fixed OOF predictions are not a formal independently
validated probability that a model wins or loses.

On an original subject-by-subject LOSO comparison:

| Model | Better | Equal | Worse |
| --- | ---: | ---: | ---: |
| kinematic_276 | 1 | 10 | 4 |
| tracking_proxy_268 | 1 | 12 | 2 |
| temporal_280 | 1 | 12 | 2 |

These numbers support a **negative result** for the added 4, 12 or
16 Bicep Curl temporal descriptors, rather than a validated gain.
Avoid reporting the GROUP5-only improvement of temporal_280
as a generalization win.

## Scientific decision: preserve original 264-feature research reference

For future *research reference only*, freeze the original M20
264-feature clip schema and M30 RandomForest specification:
300 trees, min_samples_leaf=2, class_weight balanced,
random_state=42, n_jobs=1, threshold=0.5. This does not
serialize, train or deploy a production model.

M32 adds a **local, privacy-safe research card generator**.
It consumes the existing M31 aggregate JSON and outputs a Markdown
table of paired Group5 and LOSO metrics, uncertainty intervals
and source camera-view slices for the retained reference model.

It rejects M31 reports with unexpected clip counts, people,
feature sets, missing uncertainty, unverified M30 reproduction,
holdout usage, or missing per-view groups. No training,
feature extraction, model deployment, source ZIP opening or
video-to-person mapping takes place in M32.

### Run on Ubuntu

```bash
cd ~/Desktop/Gym-Pose-Detection-System
git pull --ff-only
source .venv-m20/bin/activate
python -m scripts.m32_curl_evidence_card
```

Result remains in a Git-ignored local path:

```text
data/processed/m32_bicep_curl_research_card.md
```

Open it:

```bash
sed -n '1,180p' data/processed/m32_bicep_curl_research_card.md
```

**Do not commit** locally sourced video, subject metadata,
annotation-level CSV or private results. The generated card
contains only aggregate statistics, without identities,
individual predictions or raw landmarks.

## What would change our conclusion?

The original Gym7 dataset's *publisher* labels classify
whole videos as good/bad, NOT shoulder/elbow diagnoses.
A 60% source-subject LO(S)O Balanced Accuracy is not a
trustworthy public technique-quality signal.

Further iteration on just these 15 subjects risks post-hoc
overfitting. A more meaningful next study needs a documented,
independently recruited group of adults, stable independent
human annotations, a priori participant-heldout split,
and consistent camera/media estimator settings. Keep
original M20 TEST untouched until one final locked protocol.

The current public website may show geometry-only previews,
but neither this card nor M30/M31 justifies turning on
automatic correct/wrong Bicep Curl scoring or safety advice.
