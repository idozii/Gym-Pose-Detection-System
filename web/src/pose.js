/** Pure pose visibility helpers. Not model inference or exercise-form diagnosis. */
export const JOINT_NAMES = Object.freeze({
  11: 'Left shoulder', 12: 'Right shoulder',
  13: 'Left elbow', 14: 'Right elbow',
  15: 'Left wrist', 16: 'Right wrist',
  23: 'Left hip', 24: 'Right hip',
  25: 'Left knee', 26: 'Right knee',
  27: 'Left ankle', 28: 'Right ankle'
});

export const TASKS = Object.freeze({
  squat: Object.freeze({
    label: 'Squat', subtitle: 'Lower body · 8 required joints',
    targets: Object.freeze([
      Object.freeze({ name: 'Knees Forward', source: 'error_knees_forward',
        joints: Object.freeze([11, 12, 23, 24, 25, 26, 27, 28]) }),
      Object.freeze({ name: 'Knees Inward', source: 'error_knees_inward',
        joints: Object.freeze([11, 12, 23, 24, 25, 26, 27, 28]) })
    ])
  }),
  ohp: Object.freeze({
    label: 'Overhead Press', subtitle: 'Upper + lower body · task-specific',
    targets: Object.freeze([
      Object.freeze({ name: 'Elbows', source: 'error_elbows',
        joints: Object.freeze([11, 12, 13, 14, 15, 16, 23, 24]) }),
      Object.freeze({ name: 'Knees', source: 'error_knees',
        joints: Object.freeze([11, 12, 23, 24, 25, 26, 27, 28]) })
    ])
  }),
  barbell_row: Object.freeze({
    label: 'Barbell Row', subtitle: 'Torso4 · shoulders and hips',
    targets: Object.freeze([
      Object.freeze({ name: 'Torso Angle', source: 'torso_angle',
        joints: Object.freeze([11, 12, 23, 24]) }),
      Object.freeze({ name: 'Lumbar Label', source: 'lumbar',
        joints: Object.freeze([11, 12, 23, 24]) })
    ])
  })
});

export const VISIBILITY_THRESHOLD = 0.5;

export function requiredJointIndices(exercise) {
  const task = TASKS[exercise];
  if (!task) throw new Error('Unknown exercise');
  return [...new Set(task.targets.flatMap((target) => target.joints))].sort((a, b) => a - b);
}

export function jointVisibility(landmarks, index, cutoff = VISIBILITY_THRESHOLD) {
  if (!Array.isArray(landmarks) || landmarks.length !== 33) return 'NO POSE';
  const p = landmarks[index];
  if (!p || ![p.x, p.y, p.z, p.visibility].every((v) => Number.isFinite(v))
    || p.visibility < 0 || p.visibility > 1) return 'UNKNOWN';
  return p.visibility >= cutoff ? 'VISIBLE' : 'LOW';
}

export function readiness(exercise, landmarks) {
  const ids = requiredJointIndices(exercise);
  const entries = ids.map((id) => ({
    id, name: JOINT_NAMES[id], state: jointVisibility(landmarks, id),
    visibility: landmarks?.[id]?.visibility
  }));
  return {
    entries,
    visibleCount: entries.filter((joint) => joint.state === 'VISIBLE').length,
    total: entries.length,
    missing: entries.filter((joint) => joint.state !== 'VISIBLE').map((joint) => joint.name),
    // Basic visibility only; not equivalent to Python source feature acceptance.
    basicVisibilityReady: entries.every((joint) => joint.state === 'VISIBLE')
  };
}

export function taskReadiness(exercise, landmarks) {
  if (!TASKS[exercise]) throw new Error('Unknown exercise');
  return TASKS[exercise].targets.map((target) => ({
    source: target.source,
    name: target.name,
    basicVisibilityReady: target.joints.every(
      (id) => jointVisibility(landmarks, id) === 'VISIBLE'
    )
  }));
}
