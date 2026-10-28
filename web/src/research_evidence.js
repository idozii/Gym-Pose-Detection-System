/**
 * W5 public research evidence snapshot (frozen reports, no raw data).
 *
 * Numbers transcribed from owner-confirmed M46/M47/M48 evidence and M50-M52
 * TRAIN integrity audit summaries. This module DOES NOT load archived videos,
 * hidden private reports, browser camera output, or publisher TEST labels.
 *
 * The 200ms/400ms evaluations are from the same offline Python MediaPipe
 * Solutions pose trace, NOT live browser Tasks Vision accuracy. At most
 * filenames, not participant/session/view identity, are disjoint.
 */
export const RESEARCH_EVIDENCE_VERSION = 'w5-frozen-train-evidence-v1';

export const EVIDENCE = Object.freeze({
  cohorts: Object.freeze({
    development: Object.freeze({
      id: 'development',
      title: 'Development TRAIN',
      sampleDescription: 'Original hash ranks 1-20',
      clips: 20,
      groundTruthReps: 312,
      independence: 'Used during detector development; not an independent validation set.',
      conditions: Object.freeze({
        200: Object.freeze({
          m39: Object.freeze({tp:149,fp:11,fn:163,mae:7.9}),
          m45: Object.freeze({tp:187,fp:15,fn:125,mae:5.8})
        }),
        400: Object.freeze({
          m39: Object.freeze({tp:118,fp:3,fn:194,mae:9.55}),
          m45: Object.freeze({tp:155,fp:8,fn:157,mae:7.45})
        })
      })
    }),
    replication: Object.freeze({
      id: 'replication',
      title: 'Filename-disjoint TRAIN replication',
      sampleDescription: 'Preselected hash ranks 21-40',
      clips: 20,
      groundTruthReps: 209,
      independence: 'Filenames only; subject/session/view independence not established.',
      conditions: Object.freeze({
        200: Object.freeze({
          m39: Object.freeze({tp:117,fp:7,fn:92,mae:4.45}),
          m45: Object.freeze({tp:129,fp:8,fn:80,mae:3.8})
        }),
        400: Object.freeze({
          m39: Object.freeze({tp:94,fp:6,fn:115,mae:5.45}),
          m45: Object.freeze({tp:121,fp:6,fn:88,mae:4.2})
        })
      })
    })
  }),
  bootstrap: Object.freeze({
    scope: 'Exactly 20 frozen replication TRAIN source videos; paired video-level bootstrap',
    draws: 5000,
    seed: 2026,
    200: Object.freeze({
      deltaF1: Object.freeze({value:0.042962,lower:0.001847,upper:0.084923}),
      deltaMAE: Object.freeze({value:-0.65,lower:-1.25,upper:-0.1}),
      deltaFalsePositivesPerVideo: Object.freeze({value:0.05,lower:-0.2,upper:0.3})
    }),
    400: Object.freeze({
      deltaF1: Object.freeze({value:0.111824,lower:0.031327,upper:0.190676}),
      deltaMAE: Object.freeze({value:-1.25,lower:-2.45,upper:-0.2}),
      deltaFalsePositivesPerVideo: Object.freeze({value:0,lower:-0.2,upper:0.25})
    })
  }),
  trainIntegrity: Object.freeze({
    annotatedClips: 40,
    groundTruthReps: 521,
    m50TimingOrDecodeFlags: 0,
    m51TarByteIdenticalClips: 40,
    m52DescriptivePtsFlags: 0,
    m52MaxApproxClockDriftMs: 0.0005,
    publisherOriginIndependentlyVerified: false,
    participantIdentityVerified: false,
    officialTestRun: false,
    remoteModelParityVerified: false
  }),
  sourceDocuments: Object.freeze({
    m46: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M46_FROZEN_UNUSED_TRAIN_REPLICATION.md',
    m47: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M47_FROZEN_TRAIN_CLIP_BOOTSTRAP.md',
    m48: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M48_FROZEN_TRAIN_EVIDENCE_PROTOCOL.md',
    m50: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M50_TRAIN_LABEL_VIDEO_INTEGRITY.md',
    m51: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M51_TRAIN_ARCHIVE_SOURCE_PARITY.md',
    m52: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M52_TRAIN_PTS_TIMELINE_AUDIT.md',
    m45Rule: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/web/scripts/m45_compare_train_relative.js',
    reproduction: 'https://github.com/idozii/Gym-Pose-Detection-System/blob/main/docs/M49_PRETEST_GIT_PROVENANCE.md'
  })
});

/**
 * M53/W6 owner-run TRAIN replication console results, 2026-10-28.
 * Values are TRANSCRIBED from the owner's local M53/W6 output and are
 * NOT independently checked against original owner-local report JSON here.
 * Both methods use exactly 20 filename-disjoint TRAIN videos (209 GT).
 * New M53 5000-draw percentile bootstrap CI bounds are owner-provided
 * console output of the owner-local report (not independently inspected
 * original JSON). No independent publisher or subject-disjoint inference.
 * W6 direct/gated zero disagreements are identical-input ALGORITHM
 * events, NOT browser MediaPipe Tasks Vision vs Python pose parity.
 */
export const OWNER_M53_W6 = Object.freeze({
  evidenceStatus: 'owner-reported-original-local-train-replay',
  sourceCohort: 'replication',
  sourceClipCount: 20,
  groundTruthReps: 209,
  newM53BootstrapIntervalsIndependentlyReviewed: false,
  ownerSharedM53BootstrapIntervals: true,
  pairedVideoBootstrap: Object.freeze({
    draws: 5000,
    seed: 2026,
    200: Object.freeze({
      normalizedMAE: Object.freeze({
        delta: -0.08404923507864687,
        lower: -0.1872277661064424,
        upper: 0.015238095238095165
      }),
      obo: Object.freeze({
        delta: 0.09999999999999998,
        lower: -0.09999999999999998,
        upper: 0.30000000000000004
      }),
      exact: Object.freeze({
        delta: 0.09999999999999998,
        lower: 0,
        upper: 0.25
      }),
      absoluteMAE: Object.freeze({
        delta: -0.6500000000000004,
        lower: -1.25,
        upper: -0.10000000000000009
      })
    }),
    400: Object.freeze({
      normalizedMAE: Object.freeze({
        delta: -0.05592170471427438,
        lower: -0.17944823455698064,
        upper: 0.07934227324533022
      }),
      obo: Object.freeze({
        delta: 0.10000000000000003,
        lower: -0.09999999999999998,
        upper: 0.30000000000000004
      }),
      exact: Object.freeze({
        delta: 0.05000000000000002,
        lower: -0.1,
        upper: 0.2
      }),
      absoluteMAE: Object.freeze({
        delta: -1.25,
        lower: -2.45,
        upper: -0.20000000000000018
      })
    })
  }),
  publisherTestEvaluated: false,
  actualBrowserPoseModelParityVerified: false,
  conditions: Object.freeze({
    200: Object.freeze({
      low_to_high: Object.freeze({
        normalizedMAE: 0.4940229816623005,
        obo: 0.4,
        exact: 0.2,
        absoluteMAE: 4.45
      }),
      relative_excursion: Object.freeze({
        normalizedMAE: 0.40997374658365365,
        obo: 0.5,
        exact: 0.3,
        absoluteMAE: 3.8
      }),
      directEventDisagreements: 0,
      gatedEventDisagreements: 0,
      framesDroppedByGate: 2
    }),
    400: Object.freeze({
      low_to_high: Object.freeze({
        normalizedMAE: 0.5397661149480034,
        obo: 0.35,
        exact: 0.15,
        absoluteMAE: 5.45
      }),
      relative_excursion: Object.freeze({
        normalizedMAE: 0.48384441023372904,
        obo: 0.45,
        exact: 0.2,
        absoluteMAE: 4.2
      }),
      directEventDisagreements: 0,
      gatedEventDisagreements: 0,
      framesDroppedByGate: 0
    })
  })
});

export function selectOwnerRunM53W6(cohortId, samplingMs) {
  const ms = Number(samplingMs);
  if (!['development', 'replication'].includes(String(cohortId)) ||
      ![200, 400].includes(ms)) {
    throw new RangeError('M53/W6 available only on 200/400ms TRAIN cohorts');
  }
  if (cohortId === 'development') return null;
  const data = OWNER_M53_W6.conditions[ms];
  const originals = EVIDENCE.cohorts.replication.conditions[ms];
  for (const [name, prior] of [
    ['low_to_high', originals.m39],
    ['relative_excursion', originals.m45]
  ]) {
    const item = data[name];
    if (Math.abs(item.absoluteMAE - prior.mae) > 1e-12 ||
        !(item.normalizedMAE >= 0) ||
        ![item.obo, item.exact].every((x) => x >= 0 && x <= 1) ||
        item.exact > item.obo ||
        !Number.isInteger(item.obo * 20) ||
        !Number.isInteger(item.exact * 20)) {
      throw new Error('M53 owner-reported TRAIN count metrics conflict with frozen M46');
    }
  }
  const bootstrap = OWNER_M53_W6.pairedVideoBootstrap[ms];
  for (const [field, delta] of [
    ['normalizedMAE', data.relative_excursion.normalizedMAE -
      data.low_to_high.normalizedMAE],
    ['obo', data.relative_excursion.obo - data.low_to_high.obo],
    ['exact', data.relative_excursion.exact - data.low_to_high.exact],
    ['absoluteMAE', data.relative_excursion.absoluteMAE -
      data.low_to_high.absoluteMAE]
  ]) {
    const ci = bootstrap[field];
    if (![ci.delta,ci.lower,ci.upper].every(Number.isFinite) ||
        ci.lower > ci.upper ||
        !Number.isFinite(delta) ||
        Math.abs(ci.delta - delta) > 1e-10) {
      throw new Error('M53 bootstrap point does not reconcile with owner TRAIN metrics');
    }
  }
  return {
    ...data,
    bootstrap,
    deltaNormalizedMAE:
      data.relative_excursion.normalizedMAE - data.low_to_high.normalizedMAE,
    deltaOBO:
      data.relative_excursion.obo - data.low_to_high.obo,
    deltaExact:
      data.relative_excursion.exact - data.low_to_high.exact
  };
}

/** Micro event-matching F1, defined only for frozen offline GT events. */
export function eventMetrics({tp,fp,fn,mae}) {
  for (const count of [tp,fp,fn]) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new RangeError('Event counts must be nonnegative integers');
    }
  }
  if (!Number.isFinite(mae) || mae < 0) {
    throw new RangeError('Absolute count MAE must be finite and nonnegative');
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const denominator = 2 * tp + fp + fn;
  return {
    tp,fp,fn,mae,precision,recall,
    f1: denominator ? 2 * tp / denominator : 0,
    groundTruthEvents: tp + fn
  };
}

/** No TEST or browser live results are selectable. */
export function selectFrozenEvidence(cohortId, samplingMs) {
  const id = String(cohortId);
  const ms = Number(samplingMs);
  if (!['development','replication'].includes(id) || ![200,400].includes(ms)) {
    throw new RangeError('Only frozen 200/400ms TRAIN cohorts are selectable');
  }
  const cohort = EVIDENCE.cohorts[id];
  const methods = cohort.conditions[ms];
  const m39 = eventMetrics(methods.m39);
  const m45 = eventMetrics(methods.m45);
  if (m39.groundTruthEvents !== cohort.groundTruthReps ||
      m45.groundTruthEvents !== cohort.groundTruthReps) {
    throw new Error('W5 source aggregate does not reconcile to GT events');
  }
  return {
    cohort, samplingMs: ms,
    m39, m45,
    deltaF1: m45.f1 - m39.f1,
    deltaMAE: m45.mae - m39.mae,
    bootstrap: id === 'replication' ? EVIDENCE.bootstrap[ms] : null,
    condition: ms === 400
      ? '400ms every-other-frame subsample of the shared 200ms offline pose trace'
      : '200ms nominal offline pose sampling with Python MediaPipe Solutions'
  };
}

export function formatFixed(x, digits = 3) {
  if (!Number.isFinite(x)) throw new TypeError('W5 expected a finite aggregate');
  return x.toFixed(digits);
}
