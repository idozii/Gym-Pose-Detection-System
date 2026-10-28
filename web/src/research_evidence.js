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
      sampleDescription: 'Original hash ranks 1–20',
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
      sampleDescription: 'Preselected hash ranks 21–40',
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
