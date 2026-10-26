/** Strict opt-in research scoring: only 33 normalized landmarks, never frames. */
const EXERCISE_IDS = new Set(['squat', 'ohp', 'barbell_row']);
export const API_URL = (import.meta.env?.VITE_INFERENCE_API_URL ?? '').trim().replace(/\/$/, '');

// Exact versioned original-source models; partial inference is NOT a six-model demo.
export const EXPECTED_MODELS = Object.freeze([
  'squat/error_knees_forward',
  'squat/error_knees_inward',
  'ohp/error_elbows',
  'ohp/error_knees',
  'barbell_row/torso_angle',
  'barbell_row/lumbar'
]);

export function sixModelReady(health) {
  if (!health || health.research_only !== true ||
      health.pose_estimator_source_parity_validated !== false ||
      health.ready_for_six_detector_demo !== true ||
      health.loaded_model_count !== EXPECTED_MODELS.length ||
      health.expected_model_count !== EXPECTED_MODELS.length ||
      !health.models || typeof health.models !== 'object') return false;
  return Object.keys(health.models).length === EXPECTED_MODELS.length &&
    EXPECTED_MODELS.every((name) => health.models[name] === 'loaded_experimental_only');
}


export function landmarkPayload(exercise, landmarks) {
  if (!EXERCISE_IDS.has(exercise)) throw new Error('Unsupported exercise');
  if (!Array.isArray(landmarks) || landmarks.length !== 33) {
    throw new Error('Exactly 33 landmarks required');
  }
  const points = landmarks.map((point) => {
    if (!point) throw new Error('Missing landmark');
    const p = {
      x: point.x, y: point.y, z: point.z,
      visibility: point.visibility
    };
    if (!Object.values(p).every((value) => typeof value === 'number' && Number.isFinite(value))
      || Math.max(Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)) > 20
      || p.visibility < 0 || p.visibility > 1) {
      throw new Error('Invalid pose coordinates or visibility');
    }
    return p;
  });
  return { exercise, landmarks: points };
}

function checkAPI(base) {
  if (!base) throw new Error('Research model backend not configured');
  if (!/^https:\/\/[^/]+$/i.test(base) &&
      !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(base)) {
    throw new Error('Model API must use HTTPS (or localhost for development)');
  }
}

export function scoreState(report) {
  if (!report || typeof report !== 'object') return { status: 'abstain', text: 'UNAVAILABLE' };
  if (report.status === 'abstain') return { status: 'abstain', text: 'ABSTAIN' };
  if (report.status !== 'experimental_source_label_score') {
    return { status: 'abstain', text: 'UNAVAILABLE' };
  }
  const p = report.error_probability;
  const t = report.threshold;
  if (typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1
      || typeof t !== 'number' || !Number.isFinite(t) || t <= 0 || t >= 1) {
    return { status: 'abstain', text: 'INVALID' };
  }
  // Publisher-annotation association ONLY; not an error/safety alert.
  return {
    status: 'scored',
    text: `Source score ${(p * 100).toFixed(1)}% · T ${t.toFixed(2)}`
  };
}

export class ResearchInferenceClient {
  constructor({ apiUrl = API_URL, fetchImpl = fetch } = {}) {
    this.base = apiUrl.replace(/\/$/, '');
    this.fetch = fetchImpl;
    checkAPI(this.base);
  }

  async request(route, { body, signal } = {}) {
    const response = await this.fetch(`${this.base}${route}`, {
      method: body ? 'POST' : 'GET',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
      credentials: 'omit', cache: 'no-store', mode: 'cors', signal
    });
    if (!response.ok) throw new Error(`Backend unavailable (HTTP ${response.status})`);
    return response.json();
  }

  async health({ signal } = {}) {
    const data = await this.request('/health', { signal });
    if (!data || !data.models || data.research_only !== true ||
        data.pose_estimator_source_parity_validated !== false) {
      throw new Error('Unexpected research backend identity');
    }
    return data;
  }

  async score(exercise, landmarks, { signal } = {}) {
    const body = landmarkPayload(exercise, landmarks);
    const data = await this.request('/score', { body, signal });
    if (data?.status !== 'experimental_source_labels_only' ||
        data.exercise !== exercise || !data.targets) {
      throw new Error('Unexpected source-label response');
    }
    return data;
  }
}
