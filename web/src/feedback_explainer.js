/** M35: descriptive movement feedback using existing local M33/M34.
 * Does not produce clinical advice, safe/unsafe or correct/wrong verdicts.
 * No publisher RF scores are used to decide exercise correctness.
 * No video, identities, raw landmark arrays or network I/O are retained.
 */
import { EXERCISE_FEEDBACK } from './exercise_feedback.js';
import { REP_POLICIES } from './rep_counter.js';

const RESEARCH_EXPLANATIONS = Object.freeze({
  squat: Object.freeze({
    source: 'knee-forward / knee-inward publisher labels',
    caution: 'These source labels do not certify knee safety or ideal squat depth.'
  }),
  ohp: Object.freeze({
    source: 'elbow / knee publisher labels',
    caution: 'These source labels do not confirm bar path, stable shoulders or a safe press.'
  }),
  barbell_row: Object.freeze({
    source: 'torso-angle / lumbar publisher labels',
    caution: 'Shoulder-to-hip landmarks cannot measure lumbar spinal curvature.'
  })
});
const rounded = (value, digits = 1) => Number(value.toFixed(digits));

/** Qualifies consented research scores without interpreting them as safety. */
export function sourceEvidenceNotice(exercise, { consent = false, report = null } = {}) {
  if (!Object.hasOwn(RESEARCH_EXPLANATIONS, exercise)) return null;
  const context = RESEARCH_EXPLANATIONS[exercise];
  if (!consent) {
    return 'Source-error RF research is OFF. Camera measurements stay local. '
      + context.caution;
  }
  if (!report || report.status !== 'experimental_source_labels_only' ||
      report.exercise !== exercise || !report.targets || typeof report.targets !== 'object') {
    return 'Source-label research signal unavailable or awaiting response. '
      + context.caution;
  }
  return 'Opt-in research scores refer to original annotations ('
    + context.source + '), NOT whether this repetition was correct. '
    + 'Low source score does NOT prove good form. ' + context.caution;
}

/** Human-readable observations based exclusively on the local cycle history. */
export function explainMovement(exercise, cycle, movement, {
  cameraRunning = true, recent = []
} = {}) {
  if (!Object.hasOwn(EXERCISE_FEEDBACK, exercise)) return null;
  const cfg = EXERCISE_FEEDBACK[exercise];
  const counting = REP_POLICIES[exercise];
  if (!cameraRunning) {
    return {
      status: 'camera_offline',
      title: 'Camera offline - no movement evaluation',
      message: 'Start the camera for local measurements. No cycle or technique quality was inferred.',
      detail: cfg.camera,
      confidence: 'Unavailable'
    };
  }
  if (!cycle || !movement || cycle.tracking !== 'visible' ||
      movement.status === 'insufficient_tracking') {
    return {
      status: 'tracking_insufficient',
      title: 'Need a clearer camera view',
      message: 'Current joint measurements are unavailable. Prior observed-cycle counts do not imply a new repetition.',
      detail: cfg.camera,
      confidence: 'Insufficient tracking'
    };
  }
  if (!Number.isFinite(cycle.cycles) || cycle.cycles < 0 ||
      !Number.isFinite(movement.angle)) {
    return {
      status: 'tracking_insufficient',
      title: 'Movement observation unavailable',
      message: 'The displayed movement has no reliable current 2D joint angle.',
      detail: cfg.camera,
      confidence: 'Insufficient tracking'
    };
  }
  if (cycle.cycles === 0) {
    return {
      status: 'observing',
      title: 'Joint angle visible - building movement history',
      message: 'No complete experimental bend-and-return cycle recorded yet. '
        + 'A partial motion is not automatically an incorrect repetition.',
      detail: counting.hint,
      confidence: 'Geometry only'
    };
  }
  const tail = Array.isArray(recent) ? recent.slice(-5).filter((entry) =>
    entry && Number.isFinite(entry.durationMs) && entry.durationMs > 0 &&
    Number.isFinite(entry.amplitudeDeg) && entry.amplitudeDeg >= 0
  ) : [];
  if (cycle.cycles < 3 || tail.length < 3) {
    const time = Number.isFinite(cycle.meanDurationSec)
      ? rounded(cycle.meanDurationSec) + ' seconds' : 'not available';
    return {
      status: 'early_cycles',
      title: cycle.cycles === 1 ? 'First movement cycle observed'
        : String(cycle.cycles) + ' movement cycles observed',
      message: 'Mean recorded cycle duration: ' + time
        + '. More continuous cycles are needed to compare rhythm within this set.',
      detail: 'These are angle-derived cycle observations, not validated repetitions or good/bad decisions.',
      confidence: 'Descriptive only'
    };
  }

  const durations = tail.map((entry) => entry.durationMs / 1000);
  const minTime = Math.min(...durations);
  const maxTime = Math.max(...durations);
  const average = durations.reduce((sum, value) => sum + value, 0) / durations.length;
  const spreadRatio = (maxTime - minTime) / average;
  const amplitude = tail.reduce((sum, entry) => sum + entry.amplitudeDeg, 0) / tail.length;
  // Heuristic threshold for descriptive variation, not a prescribed tempo.
  const comparison = spreadRatio >= 0.3
    ? 'Recorded cycle durations vary within this short set.'
    : 'Recorded cycle durations are relatively similar in this short set.';
  return {
    status: 'cycle_comparison',
    title: String(cycle.cycles) + ' movement cycles observed',
    message: comparison + ' Last ' + String(tail.length) + ' measured cycles took '
      + rounded(minTime) + '-' + rounded(maxTime)
      + ' seconds each. Mean projected joint-angle excursion was '
      + rounded(amplitude, 0) + ' degrees.',
    detail: 'This is a relative within-set observation, NOT a technique score or recommended tempo. '
      + cfg.limitation,
    confidence: 'Descriptive only'
  };
}
