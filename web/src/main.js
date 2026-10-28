import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { TASKS, requiredJointIndices, readiness, taskReadiness } from './pose.js';
import { API_URL, ResearchInferenceClient, scoreState, sixModelReady } from './inference.js';
import { CAMERA_FACING, cameraConstraints, mirrorPreview, cameraFallbackMessage } from './camera.js';
import { PushupMotionWindow } from './pushup.js';
import { ExerciseFeedbackWindow, EXERCISE_FEEDBACK } from './exercise_feedback.js';
import { RepCycleCounter, REP_POLICIES } from './rep_counter.js';
import { LiveRelativeExcursionCounter, RESEARCH_DEMO_MODES, isResearchSquat } from './live_relative_excursion.js';
import { cameraErrorMessage, shouldRevokeOnHidden, shouldIgnoreEndedTrack } from './camera_readiness.js';
import { SessionDashboard, tempoBars, tempoFromM34, tempoFromM45 } from './session_dashboard.js';
import { explainMovement, sourceEvidenceNotice } from './feedback_explainer.js';
import './style.css';

const ASSET_VERSION = '1.1.1';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ASSET_VERSION}/wasm`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const state = {
  exercise: 'squat', facingMode: CAMERA_FACING.FRONT, mirrorSelfie: true,
  running: false, starting: false, token: 0,
  stream: null, landmarker: null, raf: null, frames: 0,
  poseFrames: 0, fps: 0, lastFrameClock: 0, lastVideoTime: -1,
  lastRendered: 0, landmarks: null,
  inferenceEnabled: false, inferenceClient: null,
  inferenceStatus: 'NOT CONNECTED', inferred: null,
  inferenceSession: 0, inferenceAbort: null,
  inferenceInFlight: false, lastInferenceAt: 0,
  pushupMotion: new PushupMotionWindow(), lastPushupPreview: null,
  exerciseFeedback: new ExerciseFeedbackWindow(), lastExerciseFeedback: null,
  repCounter: new RepCycleCounter(), lastRepAnalysis: null,
  liveRelative: new LiveRelativeExcursionCounter(), lastM45: null,
  demoMode: 'compare', sessionDashboard: new SessionDashboard()
};

document.querySelector('#app').innerHTML = `
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="${import.meta.env.BASE_URL}" aria-label="Gym Pose Lab home">
        <span class="brand-logo"><span class="brand-mark">G</span><span class="brand-spark">✳</span></span>
        <span>GYM POSE <strong>LAB</strong></span>
      </a>
      <div class="top-right">
        <span class="top-separator"></span>
        <span class="version"><span class="version-dot"></span> RESEARCH MODE / V0.5</span>
        <a class="github-link" href="https://github.com/idozii/Gym-Pose-Detection-System" target="_blank" rel="noopener noreferrer">View source ↗</a>
      </div>
    </header>

    <main>
      <section class="intro">
        <div class="eyebrow"><span class="eyebrow-line"></span> MOVEMENT INTELLIGENCE / LAB 01</div>
        <h1>Every rep.<br><em>Another perspective.</em></h1>
        <p>Turn your phone into a private movement lab. Observe camera-plane motion with an experimental cycle counter for Squat, OHP and Barbell Row. Push-up remains a local geometry preview; all model scores remain separate.</p>
        <div class="hero-pills"><span><i></i> ON-DEVICE POSE TRACKING</span><span>04 EXERCISES</span><span>06 SOURCE MODELS / PUSH-UP LOCAL ONLY</span></div>
      </section>

      <section class="workbench" aria-label="Pose detection workbench">
        <div class="video-area">
          <div class="video-bar">
            <div class="bar-left"><span class="lens-dot"></span> MOVEMENT STUDIO <span class="camera-mode-indicator" id="cameraModeIndicator">FRONT / MIRRORED</span></div>
            <span class="runtime-label" id="runtimeStatus">CAMERA OFFLINE</span>
          </div>
          <div class="camera-stage">
            <video id="sourceVideo" muted playsinline autoplay aria-hidden="true"></video>
            <canvas id="poseCanvas" width="960" height="720" aria-label="Live camera with pose landmark overlay" role="img"></canvas>
            <div class="stage-placeholder" id="placeholder">
              <div class="placeholder-glyph" aria-hidden="true"><span>✳</span></div>
              <div class="placeholder-title">Your studio starts here.</div>
              <p>Choose front camera for a mirrored selfie view,<br>or rear camera to observe someone else training.</p>
              <div class="placeholder-faint">PRIVATE PREVIEW / NO VIDEO UPLOAD</div>
            </div>
            <div class="live-indicator" id="liveIndicator" hidden><span class="red-dot"></span> LIVE</div>
            <div class="video-footer" id="videoFooter" hidden>
              <span>POSE LANDMARKER · ON DEVICE</span><span id="videoStatus">Scanning for pose…</span>
            </div>
          </div>
          <div class="controls">
            <div class="camera-actions">
              <button class="primary-btn" type="button" id="cameraButton"><span aria-hidden="true">◉</span> Start camera</button>
              <label class="camera-picker" for="cameraFacing"><span>CAMERA</span>
                <select id="cameraFacing" aria-label="Choose front or rear camera">
                  <option value="user">Front / Selfie</option>
                  <option value="environment">Rear / Observer</option>
                </select>
              </label>
              <button type="button" class="mirror-btn" id="mirrorButton" aria-pressed="true" aria-label="Disable mirrored front camera preview">⇋ <span>Mirror on</span></button>
            </div>
            <span class="control-hint" id="controlHint" role="status" aria-live="polite">Use HTTPS, allow camera access, then choose your angle. No recording.</span>
          </div>
        </div>

        <aside class="inspector">
          <div class="panel-header">
            <div><div class="small-label">THE CONTROL DECK</div><h2>Pose inspector<span class="title-accent">.</span></h2></div>
            <span class="outline-pill"><span class="tiny-pulse"></span> LIVE READOUT</span>
          </div>

          <div class="field-group">
            <label for="exerciseSelect" class="field-label">01 / SELECT EXERCISE</label>
            <select id="exerciseSelect" aria-label="Select exercise">
              <option value="squat">Squat</option>
              <option value="ohp">Overhead Press</option>
              <option value="barbell_row">Barbell Row</option>
              <option value="pushup">Push-up - Local Preview</option>
            </select>
            <p class="field-help" id="exerciseHelp"></p>
          </div>

          <div class="metrics-grid">
            <div class="metric"><span>POSE SEEN</span><strong id="poseMetric">-</strong></div>
            <div class="metric"><span>FPS</span><strong id="fpsMetric">-</strong></div>
          </div>

          <section class="pushup-preview" id="pushupPreview" aria-label="Push-up local movement geometry" hidden>
            <div class="preview-overline">PUSH-UP / LOCAL GEOMETRY ONLY</div>
            <div class="pushup-preview-metrics">
              <div><span>ELBOW ANGLE (2D)</span><strong id="pushupAngle">-</strong></div>
              <div><span>10S OBSERVED RANGE</span><strong id="pushupRange">-</strong></div>
            </div>
            <p id="pushupPreviewHelp">Show shoulder, elbow, wrist and hip on one side. Angle is a camera-plane estimate, not a rep count or technique assessment. No Push-up quality classifier is deployed.</p>
          </section>

          <section class="exercise-feedback" id="exerciseFeedback" aria-label="Local movement feedback" aria-live="polite">
            <div class="feedback-head"><span>LOCAL MOVEMENT FEEDBACK / NO FORM VERDICT</span><span class="feedback-state" id="feedbackState">CAMERA OFFLINE</span></div>
            <div class="feedback-metrics">
              <div><span id="feedbackMetricName">JOINT ANGLE (2D)</span><strong id="feedbackAngle">-</strong></div>
              <div><span>10-SECOND OBSERVED RANGE</span><strong id="feedbackRange">-</strong></div>
            </div>
            <p class="feedback-note" id="feedbackNote">Start your camera to observe measurable joint movement.</p>
            <p class="feedback-camera" id="feedbackCamera">Camera guidance will appear here.</p>
            <p class="feedback-limit" id="feedbackLimit">Camera-plane geometry is not correct/wrong exercise classification.</p>
          </section>

          <section class="rep-analysis" id="repAnalysis" aria-label="Experimental local movement cycle counter">
            <div class="rep-header">
              <div><div class="rep-eyebrow">M34 / EXPERIMENTAL MOVEMENT CYCLES</div><strong>Observed cycles <span class="rep-explain">(not verified reps)</span></strong></div>
              <button type="button" class="rep-reset" id="resetSet" aria-label="Reset observed cycles for current exercise">Reset set</button>
            </div>
            <div class="rep-metrics">
              <div><span>OBSERVED CYCLES</span><strong id="repCount">0</strong></div>
              <div><span>MEAN CYCLE TIME</span><strong id="repTempo">-</strong></div>
              <div><span>MEAN ANGLE EXCURSION</span><strong id="repAmplitude">-</strong></div>
            </div>
            <p class="rep-status" id="repStatus" role="status" aria-live="polite">Start the camera to observe a complete bending-and-return cycle.</p>
            <p class="rep-guidance" id="repGuidance">Only on-device camera-plane heuristics are used; this is not a verified rep count or exercise-form score.</p>
          </section>

          <section class="session-dashboard" id="sessionDashboard" aria-label="Local experimental repetition session dashboard">
            <div class="session-heading">
              <div>
                <div class="rep-eyebrow">W3 / LIVE SESSION TELEMETRY</div>
                <strong>Movement session<span class="session-subtitle"> / experimental observations only</span></strong>
              </div>
              <span class="session-live-state" id="sessionState">CAMERA OFFLINE</span>
            </div>
            <div class="session-stat-grid" aria-live="off">
              <div class="session-stat session-stat-primary"><span>SESSION ELAPSED</span><strong id="sessionElapsed">00:00</strong><small>Since camera start or Reset set</small></div>
              <div class="session-stat"><span>POSE SEEN</span><strong id="sessionPoseSeen">--</strong><small>Of analyzed video frames</small></div>
              <div class="session-stat"><span>REQUIRED JOINTS VISIBLE</span><strong id="sessionJointReady">--</strong><small>Of analyzed video frames, not confidence</small></div>
              <div class="session-stat"><span>FRAMES ANALYZED</span><strong id="sessionFrames">0</strong><small>Local pose observations</small></div>
            </div>
            <div class="session-tempo-grid">
              <div class="session-tempo-panel">
                <div class="session-tempo-heading"><span>M34 / FULL CYCLE DURATION</span><strong id="sessionM34Mean">--</strong></div>
                <div class="session-tempo-bars" id="sessionM34Bars" role="img" aria-label="M34 no completed movements yet"></div>
                <p>Up to 12 recent completed cycles, oldest to newest</p>
              </div>
              <div class="session-tempo-panel" id="sessionM45Panel">
                <div class="session-tempo-heading"><span>M45 / LOW-TO-RETURN DURATION</span><strong id="sessionM45Mean">--</strong></div>
                <div class="session-tempo-bars session-tempo-bars-m45" id="sessionM45Bars" role="img" aria-label="M45 no completed movements yet"></div>
                <p>Up to 12 recovery segments, oldest to newest</p>
              </div>
            </div>
            <p class="session-disclaimer">Tempo panels use different start/end definitions and independent scales: bar heights and values are NOT accuracy, coaching guidance or directly comparable rep times. Camera data stays local unless research inference is separately enabled.</p>
          </section>

          <section class="research-comparison" id="researchComparison" aria-label="Experimental Squat counter comparison">
            <div class="research-comparison-head">
              <div><div class="rep-eyebrow">W2 / LIVE RESEARCH DEMO</div><strong>Two independent cycle heuristics, one camera</strong></div>
              <label class="research-mode-label" for="researchMode">VIEW
                <select id="researchMode" aria-label="Choose live research comparison view">
                  <option value="compare">Compare</option>
                  <option value="m34">M34 only</option>
                  <option value="m45">M45 only</option>
                </select>
              </label>
            </div>
            <div class="research-counter-grid" aria-live="off">
              <div class="research-count-card" id="researchM34Card">
                <span>M34 / ORIGINAL LIVE</span>
                <strong id="researchM34Count">0</strong>
                <small>Absolute high-low-high / 400ms sampling</small>
              </div>
              <div class="research-count-card" id="researchM45Card">
                <span>M45 / EXPERIMENTAL LIVE</span>
                <strong id="researchM45Count">0</strong>
                <small>Relative excursion / 200ms target</small>
              </div>
            </div>
            <p class="research-comparison-status" id="researchComparisonStatus" role="status" aria-live="polite">Start the camera for a Squat-only research comparison.</p>
            <p class="research-comparison-limit">The counters have different rules and sampling cadences. Their difference is NOT an accuracy score: no live GT labels exist. M45 was studied offline on publisher TRAIN with a different MediaPipe pose implementation. Not exercise form, safety, or browser parity evidence. Neither mode sends video or landmark arrays.</p>
          </section>

          <section class="feedback-explainer" id="feedbackExplainer" aria-label="Explanations of measured movement, not form grading">
            <div class="feedback-explainer-heading">M35 / EXPLAINABLE OBSERVATIONS</div>
            <strong class="feedback-explainer-title" id="explainTitle">Awaiting camera measurements</strong>
            <p class="feedback-explainer-message" id="explainMessage">Feedback uses only observed camera-plane geometry and the local cycle history.</p>
            <p class="feedback-explainer-detail" id="explainDetail">No correct/wrong technique evaluation is made.</p>
            <div class="feedback-research-divider">SOURCE MODEL CONTEXT / OPTIONAL RESEARCH ONLY</div>
            <p class="feedback-explainer-source" id="explainSource">Research RF model signals are separate from movement feedback.</p>
          </section>

          <div class="section-divider"></div>
          <div class="section-heading"><div>02 / JOINT VISIBILITY</div><span id="jointCount">0/8</span></div>
          <div class="progress-track"><div class="progress-fill" id="jointProgress"></div></div>
          <div class="joint-list" id="jointList"></div>
          <p class="hint-note" id="framingHint">Keep shoulders, hips, knees and ankles visible. Visibility is not technique quality.</p>

          <div class="section-divider"></div>
          <div class="section-heading"><div id="sourceHeading">03 / SOURCE-LABEL MODELS</div></div>
          <div class="inference-consent">
            <label class="consent-label" for="enableInference">
              <input type="checkbox" id="enableInference" />
              <span><strong>Enable research inference</strong><small>Send only 33 pose landmarks to the configured Python API (max 2 requests/sec). Webcam images never leave this page.</small></span>
            </label>
            <p class="inference-state" id="inferenceState">MODEL SERVER NOT CONFIGURED</p>
            <p class="preview-inference-help" id="inferenceInfo"></p>
          </div>
          <div id="models" class="models"></div>
          <div class="notice">
            <div class="notice-icon" aria-hidden="true">!</div>
            <div><strong>Research preview / not coaching</strong>
              <p>Only original source-annotation scores from operator-trusted RF models may appear, after consent. The Python models are not bundled with this website. The browser pose estimator differs from the source training pipeline; results have <strong>unvalidated domain shift</strong> and cannot determine safe or correct technique.</p>
            </div>
          </div>
        </aside>
      </section>
      <footer class="site-footer">
        <span>GYM POSE LAB <strong>©</strong> MOVEMENT RESEARCH</span>
        <p>Camera video runs in your browser. If you separately enable research inference, only 33 pose landmarks are transmitted to the configured backend. No video uploading or automated technique and injury advice. MediaPipe assets are fetched from Google/CDN.</p>
      </footer>
    </main>
  </div>
`;

const ui = {
  video: document.querySelector('#sourceVideo'),
  canvas: document.querySelector('#poseCanvas'),
  button: document.querySelector('#cameraButton'),
  cameraFacing: document.querySelector('#cameraFacing'),
  mirrorButton: document.querySelector('#mirrorButton'),
  cameraModeIndicator: document.querySelector('#cameraModeIndicator'),
  exercise: document.querySelector('#exerciseSelect'),
  help: document.querySelector('#exerciseHelp'),
  models: document.querySelector('#models'),
  jointList: document.querySelector('#jointList'),
  jointCount: document.querySelector('#jointCount'),
  jointProgress: document.querySelector('#jointProgress'),
  framingHint: document.querySelector('#framingHint'),
  placeholder: document.querySelector('#placeholder'),
  indicator: document.querySelector('#liveIndicator'),
  videoFooter: document.querySelector('#videoFooter'),
  status: document.querySelector('#runtimeStatus'),
  videoStatus: document.querySelector('#videoStatus'),
  hint: document.querySelector('#controlHint'),
  poseMetric: document.querySelector('#poseMetric'),
  fpsMetric: document.querySelector('#fpsMetric'),
  inferenceToggle: document.querySelector('#enableInference'),
  inferenceState: document.querySelector('#inferenceState'),
  inferenceInfo: document.querySelector('#inferenceInfo'),
  sourceHeading: document.querySelector('#sourceHeading'),
  pushupPreview: document.querySelector('#pushupPreview'),
  pushupAngle: document.querySelector('#pushupAngle'),
  pushupRange: document.querySelector('#pushupRange'),
  pushupPreviewHelp: document.querySelector('#pushupPreviewHelp'),
  exerciseFeedback: document.querySelector('#exerciseFeedback'),
  feedbackMetricName: document.querySelector('#feedbackMetricName'),
  feedbackAngle: document.querySelector('#feedbackAngle'),
  feedbackRange: document.querySelector('#feedbackRange'),
  feedbackState: document.querySelector('#feedbackState'),
  feedbackNote: document.querySelector('#feedbackNote'),
  feedbackCamera: document.querySelector('#feedbackCamera'),
  feedbackLimit: document.querySelector('#feedbackLimit'),
  repAnalysis: document.querySelector('#repAnalysis'),
  repCount: document.querySelector('#repCount'),
  repTempo: document.querySelector('#repTempo'),
  repAmplitude: document.querySelector('#repAmplitude'),
  repStatus: document.querySelector('#repStatus'),
  repGuidance: document.querySelector('#repGuidance'),
  resetSet: document.querySelector('#resetSet'),
  sessionDashboard: document.querySelector('#sessionDashboard'),
  sessionState: document.querySelector('#sessionState'),
  sessionElapsed: document.querySelector('#sessionElapsed'),
  sessionPoseSeen: document.querySelector('#sessionPoseSeen'),
  sessionJointReady: document.querySelector('#sessionJointReady'),
  sessionFrames: document.querySelector('#sessionFrames'),
  sessionM34Mean: document.querySelector('#sessionM34Mean'),
  sessionM34Bars: document.querySelector('#sessionM34Bars'),
  sessionM45Panel: document.querySelector('#sessionM45Panel'),
  sessionM45Mean: document.querySelector('#sessionM45Mean'),
  sessionM45Bars: document.querySelector('#sessionM45Bars'),
  researchComparison: document.querySelector('#researchComparison'),
  researchMode: document.querySelector('#researchMode'),
  researchM34Card: document.querySelector('#researchM34Card'),
  researchM45Card: document.querySelector('#researchM45Card'),
  researchM34Count: document.querySelector('#researchM34Count'),
  researchM45Count: document.querySelector('#researchM45Count'),
  researchComparisonStatus: document.querySelector('#researchComparisonStatus'),
  feedbackExplainer: document.querySelector('#feedbackExplainer'),
  explainTitle: document.querySelector('#explainTitle'),
  explainMessage: document.querySelector('#explainMessage'),
  explainDetail: document.querySelector('#explainDetail'),
  explainSource: document.querySelector('#explainSource')
};

function updateCameraControls() {
  const front = state.facingMode === CAMERA_FACING.FRONT;
  ui.cameraFacing.value = state.facingMode;
  ui.mirrorButton.disabled = !front;
  ui.mirrorButton.setAttribute('aria-pressed', String(front && state.mirrorSelfie));
  ui.mirrorButton.setAttribute('aria-label', front
    ? (state.mirrorSelfie ? 'Disable mirrored selfie preview' : 'Enable mirrored selfie preview')
    : 'Rear camera preview is never mirrored');
  ui.mirrorButton.querySelector('span').textContent = front
    ? (state.mirrorSelfie ? 'Mirror on' : 'Mirror off') : 'Rear unmirrored';
  ui.cameraModeIndicator.textContent = front
    ? (state.mirrorSelfie ? 'FRONT / MIRRORED' : 'FRONT / NATURAL') : 'REAR / OBSERVER';
}

function setStatus(text, hint) {
  ui.status.textContent = text;
  if (hint) ui.hint.textContent = hint;
}

function renderModels() {
  const isPushup = state.exercise === 'pushup';
  ui.inferenceToggle.disabled = isPushup || !API_URL;
  ui.inferenceState.textContent = isPushup
    ? 'PUSH-UP LOCAL ONLY / NO MODEL SCORE'
    : !API_URL ? 'MODEL SERVER NOT CONFIGURED' : state.inferenceStatus;
  ui.inferenceInfo.textContent = isPushup
    ? 'Push-up camera geometry is on-device only. The experimental Gym7 good/bad classifiers are not deployed because external validation did not establish reliability.'
    : 'Only source-label research scoring is eligible for the remote API after consent.';
  ui.sourceHeading.textContent = isPushup ? '03 / PUSH-UP MODEL STATUS' : '03 / SOURCE-LABEL MODELS';
  if (isPushup) {
    ui.models.innerHTML = '<div class="model-row"><div><strong>Push-up quality model</strong><span>Gym7 whole-video good/bad research - not externally validated, no live temporal model</span></div><span class="offline-tag">NOT DEPLOYED</span></div>';
    return;
  }
  const targets = TASKS[state.exercise].targets;
  ui.inferenceState.textContent = !API_URL ? 'MODEL SERVER NOT CONFIGURED'
    : state.inferenceStatus;
  ui.models.innerHTML = targets.map((target) => {
    const report = state.inferenceEnabled ? state.inferred?.targets?.[target.source] : null;
    const stateView = report ? scoreState(report) : null;
    const label = stateView
      ? stateView.text
      : !API_URL ? 'NOT DEPLOYED'
        : state.inferenceEnabled ? 'AWAITING SCORE' : 'CONSENT REQUIRED';
    const klass = stateView?.status === 'scored' ? 'source-tag' : 'offline-tag';
    return `
      <div class="model-row">
        <div><strong>${target.name}</strong><span>${target.source}</span></div>
        <span class="${klass}">${label}</span>
      </div>
    `;
  }).join('');
}

function drawSessionTempo(container, seconds, label) {
  const bars = tempoBars(seconds);
  container.replaceChildren();
  container.setAttribute('aria-label',
    bars.length
      ? label + ' durations in seconds, oldest to newest: ' +
        bars.map((bar) => bar.durationSec.toFixed(2)).join(', ')
      : label + ': no completed movements yet');
  if (bars.length === 0) {
    const note = document.createElement('span');
    note.className = 'session-tempo-empty';
    note.textContent = 'NO COMPLETED CYCLES YET';
    container.append(note);
    return;
  }
  for (const bar of bars) {
    const element = document.createElement('span');
    element.className = 'session-tempo-bar';
    element.style.height = bar.heightPercent + '%';
    element.title = label + ' segment ' + bar.index + ': ' +
      bar.durationSec.toFixed(2) + ' seconds';
    element.setAttribute('aria-hidden', 'true');
    container.append(element);
  }
}

function updateSessionDashboard() {
  const isPushup = state.exercise === 'pushup';
  ui.sessionDashboard.hidden = isPushup;
  if (isPushup) return;
  const data = state.sessionDashboard.summary(performance.now());
  ui.sessionElapsed.textContent = data.elapsedLabel;
  ui.sessionPoseSeen.textContent = data.poseSeenPct === null
    ? '--' : data.poseSeenPct + '%';
  ui.sessionJointReady.textContent = data.requiredReadyPct === null
    ? '--' : data.requiredReadyPct + '%';
  ui.sessionFrames.textContent = String(data.sampleFrames);
  ui.sessionState.textContent = data.status;
  ui.sessionState.dataset.status = data.status;
  const m34 = state.running ? state.lastRepAnalysis : null;
  ui.sessionM34Mean.textContent = m34?.meanDurationSec == null
    ? '--' : m34.meanDurationSec.toFixed(1) + ' s avg';
  drawSessionTempo(ui.sessionM34Bars, tempoFromM34(state.repCounter.recent),
    'M34 full cycle');
  const squat = isResearchSquat(state.exercise);
  ui.sessionM45Panel.hidden = !squat;
  if (squat) {
    const m45 = state.running ? state.lastM45 : null;
    ui.sessionM45Mean.textContent = m45?.meanRecoverySec == null
      ? '--' : m45.meanRecoverySec.toFixed(1) + ' s avg';
    drawSessionTempo(ui.sessionM45Bars, tempoFromM45(state.liveRelative.events),
      'M45 low to return');
  }
}

function updateInspector() {
  updateSessionDashboard();
  const data = readiness(state.exercise, state.landmarks);
  const rawPoseSeen = state.running && Array.isArray(state.landmarks);
  ui.poseMetric.textContent = state.frames === 0 ? '-' : `${Math.round(state.poseFrames / state.frames * 100)}%`;
  ui.fpsMetric.textContent = state.frames === 0 ? '-' : state.fps.toFixed(1);
  ui.jointCount.textContent = `${data.visibleCount}/${data.total}`;
  ui.jointProgress.style.width = `${data.visibleCount / data.total * 100}%`;
  ui.jointList.innerHTML = data.entries.map((joint) => {
    const label = joint.state === 'VISIBLE' ? 'VISIBLE' : joint.state === 'LOW' ? 'LOW' : joint.state === 'NO POSE' ? 'NO POSE' : 'UNKNOWN';
    const klass = joint.state === 'VISIBLE' ? 'joint-ok' : 'joint-low';
    return `<div class="joint-item"><span class="joint-name">${joint.name}</span><span class="${klass}">${label}</span></div>`;
  }).join('');
  const isPushup = state.exercise === 'pushup';
  ui.pushupPreview.hidden = !isPushup;
  ui.exerciseFeedback.hidden = isPushup;
  ui.repAnalysis.hidden = isPushup;
  ui.researchComparison.hidden = !isResearchSquat(state.exercise);
  ui.feedbackExplainer.hidden = isPushup;
  if (isResearchSquat(state.exercise)) {
    const original = state.running ? state.lastRepAnalysis : null;
    const candidate = state.running ? state.lastM45 : null;
    ui.researchMode.value = state.demoMode;
    ui.researchM34Card.hidden = state.demoMode === 'm45';
    ui.researchM45Card.hidden = state.demoMode === 'm34';
    ui.researchM34Count.textContent = String(original?.cycles ?? 0);
    ui.researchM45Count.textContent = String(candidate?.cycles ?? 0);
    ui.researchComparisonStatus.textContent = !state.running
      ? 'Camera offline. Both research cycle histories are cleared.'
      : !candidate || candidate.tracking !== 'visible'
        ? 'Tracking unavailable. Keep hip, knee and ankle visible from the side.'
        : candidate.phase === 'await_drop'
          ? 'M45 observes a high knee-angle anchor and waits for a 35-degree drop.'
          : 'M45 observed a drop and is checking recovery to at least 155 degrees.';
  }
  if (!isPushup) {
    const observation = explainMovement(
      state.exercise,
      state.running ? state.lastRepAnalysis : null,
      state.running ? state.lastExerciseFeedback : null,
      { cameraRunning: state.running, recent: state.repCounter.recent }
    );
    ui.explainTitle.textContent = observation.title;
    ui.explainMessage.textContent = observation.message;
    ui.explainDetail.textContent = observation.detail;
    ui.explainSource.textContent = sourceEvidenceNotice(state.exercise, {
      consent: state.inferenceEnabled,
      report: state.inferenceEnabled ? state.inferred : null
    });
  }
  if (!isPushup) {
    const rep = state.running ? state.lastRepAnalysis : null;
    ui.repCount.textContent = String(rep?.cycles ?? 0);
    ui.repTempo.textContent = rep?.meanDurationSec == null
      ? '-' : `${rep.meanDurationSec.toFixed(1)} s`;
    ui.repAmplitude.textContent = rep?.meanExcursionDeg == null
      ? '-' : `${rep.meanExcursionDeg.toFixed(0)}°`;
    ui.repStatus.textContent = !state.running
      ? 'Camera offline. Local cycle history is cleared when the camera stops.'
      : rep?.tracking !== 'visible' ? 'Insufficient tracking. Show all required joints before counting.'
        : rep.phase === 'await_high' ? 'Waiting for a relatively straight starting position.'
          : rep.phase === 'await_low' ? 'Waiting for clear bending of the observed joint.'
            : 'Bending observed. Waiting for return to the starting angle.';
    ui.repGuidance.textContent = REP_POLICIES[state.exercise].hint +
      ' This count is experimental, not validated repetitions, safety or technique.';
  }
  if (!isPushup) {
    const policy = EXERCISE_FEEDBACK[state.exercise];
    const feedback = state.running ? state.lastExerciseFeedback : null;
    ui.feedbackMetricName.textContent = policy.measure.toUpperCase();
    ui.feedbackAngle.textContent = feedback?.angle == null ? '-' : `${feedback.angle.toFixed(0)}°`;
    ui.feedbackRange.textContent = feedback?.range == null ? '-' : `${feedback.range.toFixed(0)}°`;
    ui.feedbackState.textContent = !state.running ? 'CAMERA OFFLINE'
      : feedback?.status === 'observing' ? 'MOVEMENT OBSERVED'
        : feedback?.status === 'collecting' ? 'COLLECTING SAMPLES' : 'NOT ENOUGH TRACKING';
    ui.feedbackNote.textContent = !state.running
      ? 'Start the camera for locally measured angles. No technique score is produced.'
      : feedback?.note ?? 'Waiting for visible pose landmarks.';
    ui.feedbackCamera.textContent = policy.camera;
    ui.feedbackLimit.textContent = policy.limitation;
  }
  if (isPushup) {
    const summary = state.lastPushupPreview;
    ui.pushupAngle.textContent = summary?.angle == null ? '-' : `${summary.angle.toFixed(0)}°`;
    ui.pushupRange.textContent = summary?.range == null ? '-' : `${summary.range.toFixed(0)}°`;
    ui.pushupPreviewHelp.textContent = summary?.angle == null
      ? 'Insufficient visible elbow landmarks. Move into frame; no estimate is shown.'
      : `Visible ${summary.side?.toLowerCase() ?? 'one'} arm - ${summary.count} local samples in 10 seconds. No rep count, correct/wrong verdict or injury advice.`;
  }
  ui.framingHint.textContent = !rawPoseSeen
    ? 'Show yourself from head to feet for Squat or OHP. This is a visibility guide only.'
    : data.basicVisibilityReady
      ? 'Required joints appear visible. This is NOT an exercise-form or model inference result.'
      : `Framing hint: ${data.missing.slice(0, 3).join(', ')}${data.missing.length > 3 ? '…' : ''}. Move so these joints are visible.`;
  const perTask = taskReadiness(state.exercise, state.landmarks);
  const n = perTask.filter((t) => t.basicVisibilityReady).length;
  ui.videoStatus.textContent = isPushup
    ? (rawPoseSeen ? 'LOCAL PUSH-UP GEOMETRY / NO QUALITY SCORE' : 'No pose detected')
    : rawPoseSeen
    ? `${n}/${perTask.length} visibility gates · ${state.inferenceEnabled ? 'research inference opt-in' : 'no remote inference'}`
    : 'No pose detected';
  renderModels();
}

async function requestResearchInference(timestamp) {
  if (state.exercise === 'pushup' || !state.inferenceEnabled || !state.inferenceClient ||
      !state.running || !Array.isArray(state.landmarks) ||
      state.inferenceInFlight || timestamp - state.lastInferenceAt < 500) return;
  state.lastInferenceAt = timestamp;
  const sequence = state.inferenceSession;
  const exercise = state.exercise;
  const points = state.landmarks;
  const controller = new AbortController();
  state.inferenceAbort = controller;
  state.inferenceInFlight = true;
  try {
    const response = await state.inferenceClient.score(exercise, points, {
      signal: controller.signal
    });
    if (sequence !== state.inferenceSession || !state.running ||
        exercise !== state.exercise || !state.inferenceEnabled) return;
    state.inferred = response;
    state.inferenceStatus = 'RESEARCH SOURCE SCORES · NOT FORM ADVICE';
  } catch (error) {
    if (sequence !== state.inferenceSession || !state.inferenceEnabled) return;
    state.inferred = null;
    state.inferenceStatus = 'INFERENCE UNAVAILABLE · ABSTAIN';
  } finally {
    if (sequence === state.inferenceSession) {
      state.inferenceInFlight = false;
      state.inferenceAbort = null;
      renderModels();
    }
  }
}

function disableResearchInference() {
  state.inferenceSession += 1;
  state.inferenceAbort?.abort();
  state.inferenceAbort = null;
  state.inferenceInFlight = false;
  state.inferenceEnabled = false;
  state.inferenceClient = null;
  state.inferred = null;
  state.lastInferenceAt = 0;
  ui.inferenceToggle.checked = false;
  state.inferenceStatus = 'CONSENT REQUIRED';
  renderModels();
}

async function enableResearchInference() {
  if (state.exercise === 'pushup') {
    disableResearchInference();
    return;
  }
  if (!ui.inferenceToggle.checked) {
    disableResearchInference();
    return;
  }
  const sequence = ++state.inferenceSession;
  state.inferenceStatus = 'CHECKING TRUSTED RESEARCH BACKEND';
  renderModels();
  try {
    const client = new ResearchInferenceClient();
    const controller = new AbortController();
    state.inferenceAbort = controller;
    const health = await client.health({ signal: controller.signal });
    if (sequence !== state.inferenceSession || !ui.inferenceToggle.checked) return;
    if (!sixModelReady(health)) {
      throw new Error('All six versioned source-error models must be available');
    }
    state.inferenceClient = client;
    state.inferenceEnabled = true;
    state.inferenceStatus = '6/6 EXPERIMENTAL MODELS AVAILABLE';
    state.lastInferenceAt = 0;
  } catch (error) {
    if (sequence !== state.inferenceSession) return;
    disableResearchInference();
    state.inferenceStatus = 'BACKEND UNAVAILABLE · NO LANDMARKS SENT';
    ui.hint.textContent = 'All six trusted source models must be provisioned. Local pose visualization remains available.';
  }
  renderModels();
}

function drawFrame() {
  const { video, canvas } = ui;
  const context = canvas.getContext('2d');
  if (!context || video.readyState < 2) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 720;
  }
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.save();
  if (mirrorPreview(state.facingMode, state.mirrorSelfie)) {
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
  }
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const points = state.landmarks;
  if (!Array.isArray(points)) { context.restore(); return; }
  const connections = PoseLandmarker.POSE_CONNECTIONS ?? [];
  context.lineWidth = Math.max(2, canvas.width / 380);
  context.strokeStyle = 'rgba(163, 255, 190, .85)';
  for (const c of connections) {
    const a = points[c.start], b = points[c.end];
    if (!a || !b || (a.visibility ?? 0) < .3 || (b.visibility ?? 0) < .3) continue;
    context.beginPath();
    context.moveTo(a.x * canvas.width, a.y * canvas.height);
    context.lineTo(b.x * canvas.width, b.y * canvas.height);
    context.stroke();
  }
  const important = new Set(requiredJointIndices(state.exercise));
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if (!p || (p.visibility ?? 0) < .15) continue;
    context.beginPath();
    context.arc(p.x * canvas.width, p.y * canvas.height,
      important.has(i) ? Math.max(4, canvas.width / 160) : Math.max(2, canvas.width / 300), 0, Math.PI * 2);
    context.fillStyle = important.has(i) && (p.visibility ?? 0) < .5 ? '#fbbf87' : '#adffd1';
    context.fill();
  }
  context.restore();
}

function loop(token, timestamp) {
  if (!state.running || token !== state.token) return;
  const v = ui.video;
  try {
    if (v.readyState >= 2 && v.currentTime !== state.lastVideoTime) {
      state.lastVideoTime = v.currentTime;
      const result = state.landmarker.detectForVideo(v, timestamp);
      state.landmarks = result.landmarks?.[0] ?? null;
      if (state.exercise === 'pushup') {
        state.lastPushupPreview = state.pushupMotion.observe(state.landmarks, timestamp);
      } else {
        state.lastExerciseFeedback = state.exerciseFeedback.observe(
          state.exercise, state.landmarks, timestamp
        );
        state.lastRepAnalysis = state.repCounter.observe(
          state.exercise, state.landmarks, timestamp
        );
        if (isResearchSquat(state.exercise)) {
          state.lastM45 = state.liveRelative.observe(state.landmarks, timestamp);
        }
      }
      state.sessionDashboard.observe(timestamp, {
        poseSeen: Array.isArray(state.landmarks),
        requiredReady: readiness(state.exercise, state.landmarks).basicVisibilityReady
      });
      state.frames++;
      if (state.landmarks) state.poseFrames++;
      if (state.lastFrameClock) {
        const delta = timestamp - state.lastFrameClock;
        if (delta > 0) state.fps = state.fps ? state.fps * .85 + (1000 / delta) * .15 : (1000 / delta);
      }
      state.lastFrameClock = timestamp;
      drawFrame();
      void requestResearchInference(timestamp);
    }
    // Clock and stalled-video status keep updating even if decoded frames stop.
    if (timestamp - state.lastRendered > 220) {
      state.lastRendered = timestamp;
      updateInspector();
    }
  } catch (error) {
    stopCamera();
    setStatus('POSE ENGINE ERROR', `The local pose engine stopped: ${error.message || 'unknown error'}`);
    return;
  }
  state.raf = requestAnimationFrame((time) => loop(token, time));
}

function stopCamera() {
  disableResearchInference();
  state.pushupMotion.reset();
  state.lastPushupPreview = null;
  state.exerciseFeedback.reset();
  state.lastExerciseFeedback = null;
  state.repCounter.reset();
  state.lastRepAnalysis = null;
  state.liveRelative.reset();
  state.lastM45 = null;
  state.sessionDashboard.reset();
  state.token++;
  state.running = false;
  state.starting = false;
  if (state.raf !== null) cancelAnimationFrame(state.raf);
  state.raf = null;
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  ui.video.pause();
  ui.video.srcObject = null;
  state.landmarker?.close();
  state.landmarker = null;
  state.landmarks = null;
  state.lastVideoTime = -1;
  state.lastFrameClock = 0;
  state.frames = 0;
  state.poseFrames = 0;
  state.fps = 0;
  ui.canvas.getContext('2d')?.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  ui.placeholder.hidden = false;
  ui.indicator.hidden = true;
  ui.videoFooter.hidden = true;
  ui.button.disabled = false;
  ui.button.innerHTML = '<span aria-hidden="true">◉</span> Start camera';
  setStatus('CAMERA OFFLINE', 'No camera recording. You can start again at any time.');
  updateInspector();
}

async function startCamera() {
  if (state.starting || state.running) return;
  if (document.hidden) {
    setStatus('CAMERA PAUSED', 'Return to the visible browser tab to start your camera.');
    return;
  }
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus('CAMERA UNAVAILABLE', 'Use HTTPS or localhost and a compatible browser.');
    return;
  }
  state.starting = true;
  const token = ++state.token;
  ui.button.disabled = true;
  ui.button.textContent = 'Loading pose engine…';
  setStatus('STARTING LOCAL CAMERA', 'Downloading MediaPipe runtime/model on the first run.');
  let tempStream = null;
  let tempLandmarker = null;
  try {
    // Only begins after explicit button click. Does not transmit webcam images.
    tempStream = await navigator.mediaDevices.getUserMedia(cameraConstraints(state.facingMode));
    const actualFacing = tempStream.getVideoTracks()[0]?.getSettings?.().facingMode;
    const fallback = cameraFallbackMessage(state.facingMode, actualFacing);
    if (token !== state.token) throw new Error('Camera start cancelled');
    ui.video.srcObject = tempStream;
    await ui.video.play();
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    tempLandmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode: 'VIDEO', numPoses: 1,
      minPoseDetectionConfidence: .5,
      minPosePresenceConfidence: .5,
      minTrackingConfidence: .5
    });
    if (token !== state.token) throw new Error('Camera start cancelled');
    state.stream = tempStream;
    state.landmarker = tempLandmarker;
    // Revocation is automatic if the browser/device unexpectedly ends video.
    const startedToken = token;
    tempStream.getVideoTracks().forEach((track) => {
      track.addEventListener('ended', () => {
        if (shouldIgnoreEndedTrack({
          cancelled: !state.running, currentToken: state.token, startedToken
        })) return;
        stopCamera();
        setStatus('CAMERA DISCONNECTED',
          'The camera track ended. Reconnect the camera, then start a new local session.');
      }, { once: true });
    });
    tempStream = null;
    tempLandmarker = null;
    state.sessionDashboard.start(performance.now());
    state.running = true;
    state.starting = false;
    state.frames = 0;
    state.poseFrames = 0;
    state.lastVideoTime = -1;
    state.fps = 0;
    ui.placeholder.hidden = true;
    ui.indicator.hidden = false;
    ui.videoFooter.hidden = false;
    ui.button.disabled = false;
    ui.button.innerHTML = '<span aria-hidden="true">■</span> Stop camera';
    setStatus('LOCAL POSE ACTIVE', fallback || 'Live local preview. Research inference requires a separate opt-in.');
    state.raf = requestAnimationFrame((timestamp) => loop(token, timestamp));
  } catch (error) {
    tempStream?.getTracks().forEach((track) => track.stop());
    tempLandmarker?.close();
    // A previous camera request can resolve after a newer switch. Never stop the new stream.
    if (token !== state.token) return;
    stopCamera();
    setStatus('COULD NOT START CAMERA', cameraErrorMessage(error));
  }
}

ui.button.addEventListener('click', () => state.running ? stopCamera() : startCamera());
ui.cameraFacing.addEventListener('change', () => {
  const next = ui.cameraFacing.value;
  if (!Object.values(CAMERA_FACING).includes(next) || next === state.facingMode) return;
  const wasActive = state.running || state.starting;
  // Close the previous physical camera before opening another (required on iOS).
  // As a privacy boundary, a camera switch revokes remote inference consent.
  stopCamera();
  state.facingMode = next;
  state.mirrorSelfie = next === CAMERA_FACING.FRONT;
  updateCameraControls();
  if (wasActive) void startCamera();
});
ui.mirrorButton.addEventListener('click', () => {
  if (state.facingMode !== CAMERA_FACING.FRONT) return;
  state.mirrorSelfie = !state.mirrorSelfie;
  updateCameraControls();
  // Pixels are mirrored only in drawFrame; model coordinates stay untouched.
  if (state.running) drawFrame();
});
ui.resetSet.addEventListener('click', () => {
  state.repCounter.reset();
  state.lastRepAnalysis = null;
  state.liveRelative.reset();
  state.lastM45 = null;
  if (state.running) state.sessionDashboard.start(performance.now());
  else state.sessionDashboard.reset();
  updateInspector();
});
ui.researchMode.addEventListener('change', () => {
  if (!RESEARCH_DEMO_MODES.includes(ui.researchMode.value)) return;
  state.demoMode = ui.researchMode.value;
  updateInspector();
});
ui.inferenceToggle.disabled = !API_URL;
ui.inferenceToggle.addEventListener('change', () => { void enableResearchInference(); });
ui.exercise.addEventListener('change', () => {
  if (!TASKS[ui.exercise.value]) return;
  disableResearchInference();
  state.pushupMotion.reset();
  state.lastPushupPreview = null;
  state.exerciseFeedback.reset();
  state.lastExerciseFeedback = null;
  state.repCounter.reset();
  state.lastRepAnalysis = null;
  state.liveRelative.reset();
  state.lastM45 = null;
  if (state.running) state.sessionDashboard.start(performance.now());
  else state.sessionDashboard.reset();
  state.exercise = ui.exercise.value;
  ui.help.textContent = TASKS[state.exercise].subtitle;
  renderModels();
  updateInspector();
});
document.addEventListener('visibilitychange', () => {
  if (shouldRevokeOnHidden({
    hidden: document.hidden, running: state.running, starting: state.starting
  })) stopCamera();
});
window.addEventListener('pagehide', stopCamera);
ui.help.textContent = TASKS[state.exercise].subtitle;
updateCameraControls();
renderModels();
updateInspector();
