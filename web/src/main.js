import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { TASKS, requiredJointIndices, readiness, taskReadiness, JOINT_NAMES } from './pose.js';
import './style.css';

const ASSET_VERSION = '1.1.1';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${ASSET_VERSION}/wasm`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const state = {
  exercise: 'squat', running: false, starting: false, token: 0,
  stream: null, landmarker: null, raf: null, frames: 0,
  poseFrames: 0, fps: 0, lastFrameClock: 0, lastVideoTime: -1,
  lastRendered: 0, landmarks: null
};

document.querySelector('#app').innerHTML = `
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="Gym Pose Lab home">
        <span class="brand-logo">G<span>•</span></span>
        <span>GYM POSE <strong>LAB</strong></span>
      </a>
      <div class="top-right">
        <span class="top-separator"></span>
        <span class="version">EXPERIMENTAL WEB PREVIEW <span class="version-dot"></span> 0.1</span>
        <a class="github-link" href="https://github.com/idozii/Gym-Pose-Detection-System" target="_blank" rel="noopener noreferrer">View source ↗</a>
      </div>
    </header>

    <main>
      <section class="intro">
        <div class="eyebrow"><span class="eyebrow-line"></span> LOCAL-FIRST COMPUTER VISION</div>
        <h1>See the motion.<br><em>Understand the pose.</em></h1>
        <p>Turn on your camera to explore how MediaPipe sees your movement. View joint visibility in real time — directly in your browser.</p>
      </section>

      <section class="workbench" aria-label="Pose detection workbench">
        <div class="video-area">
          <div class="video-bar">
            <div class="bar-left"><span class="lens-dot"></span> LIVE POSE STUDIO</div>
            <span class="runtime-label" id="runtimeStatus">CAMERA OFFLINE</span>
          </div>
          <div class="camera-stage">
            <video id="sourceVideo" muted playsinline autoplay aria-hidden="true"></video>
            <canvas id="poseCanvas" width="960" height="720" aria-label="Live camera with pose landmark overlay" role="img"></canvas>
            <div class="stage-placeholder" id="placeholder">
              <div class="placeholder-glyph" aria-hidden="true">◎</div>
              <div class="placeholder-title">Your camera, your space.</div>
              <p>Enable your webcam to start local pose tracking.<br>No camera frames are saved or sent to our server.</p>
              <div class="placeholder-faint">CAMERA READY WHEN YOU ARE</div>
            </div>
            <div class="live-indicator" id="liveIndicator" hidden><span class="red-dot"></span> LIVE</div>
            <div class="video-footer" id="videoFooter" hidden>
              <span>POSE LANDMARKER · ON DEVICE</span><span id="videoStatus">Scanning for pose…</span>
            </div>
          </div>
          <div class="controls">
            <button class="primary-btn" type="button" id="cameraButton"><span aria-hidden="true">◉</span> Start camera</button>
            <span class="control-hint" id="controlHint">Requires camera permission and HTTPS or localhost.</span>
          </div>
        </div>

        <aside class="inspector">
          <div class="panel-header">
            <div><div class="small-label">WORKBENCH</div><h2>Pose inspector</h2></div>
            <span class="outline-pill">LIVE DATA</span>
          </div>

          <div class="field-group">
            <label for="exerciseSelect" class="field-label">01 / SELECT EXERCISE</label>
            <select id="exerciseSelect" aria-label="Select exercise">
              <option value="squat">Squat</option>
              <option value="ohp">Overhead Press</option>
              <option value="barbell_row">Barbell Row</option>
            </select>
            <p class="field-help" id="exerciseHelp"></p>
          </div>

          <div class="metrics-grid">
            <div class="metric"><span>POSE SEEN</span><strong id="poseMetric">—</strong></div>
            <div class="metric"><span>FPS</span><strong id="fpsMetric">—</strong></div>
          </div>

          <div class="section-divider"></div>
          <div class="section-heading"><div>02 / JOINT VISIBILITY</div><span id="jointCount">0/8</span></div>
          <div class="progress-track"><div class="progress-fill" id="jointProgress"></div></div>
          <div class="joint-list" id="jointList"></div>
          <p class="hint-note" id="framingHint">Keep shoulders, hips, knees and ankles visible. Visibility is not technique quality.</p>

          <div class="section-divider"></div>
          <div class="section-heading"><div>03 / SOURCE-LABEL MODELS</div></div>
          <div id="models" class="models"></div>
          <div class="notice">
            <div class="notice-icon" aria-hidden="true">i</div>
            <div><strong>Research preview only</strong>
              <p>Six Python Random Forest models exist in the original local project, but are <strong>not deployed</strong> in this browser build. This website does not invent scores or judge exercise form.</p>
            </div>
          </div>
        </aside>
      </section>
      <footer class="site-footer">
        <span>GYM POSE LAB / EXPERIMENTAL RESEARCH SOFTWARE</span>
        <p>Pose runs in your browser. Camera access is optional and can be stopped anytime. MediaPipe runtime/model files are fetched from Google/CDN when you start. No automatic injury, safety, or technique advice.</p>
      </footer>
    </main>
  </div>
`;

const ui = {
  video: document.querySelector('#sourceVideo'),
  canvas: document.querySelector('#poseCanvas'),
  button: document.querySelector('#cameraButton'),
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
  fpsMetric: document.querySelector('#fpsMetric')
};

function setStatus(text, hint) {
  ui.status.textContent = text;
  if (hint) ui.hint.textContent = hint;
}

function renderModels() {
  const targets = TASKS[state.exercise].targets;
  ui.models.innerHTML = targets.map((target) => `
    <div class="model-row">
      <div><strong>${target.name}</strong><span>${target.source}</span></div>
      <span class="offline-tag">NOT DEPLOYED</span>
    </div>
  `).join('');
}

function updateInspector() {
  const data = readiness(state.exercise, state.landmarks);
  const rawPoseSeen = state.running && Array.isArray(state.landmarks);
  ui.poseMetric.textContent = state.frames === 0 ? '—' : `${Math.round(state.poseFrames / state.frames * 100)}%`;
  ui.fpsMetric.textContent = state.frames === 0 ? '—' : state.fps.toFixed(1);
  ui.jointCount.textContent = `${data.visibleCount}/${data.total}`;
  ui.jointProgress.style.width = `${data.visibleCount / data.total * 100}%`;
  ui.jointList.innerHTML = data.entries.map((joint) => {
    const label = joint.state === 'VISIBLE' ? 'VISIBLE' : joint.state === 'LOW' ? 'LOW' : joint.state === 'NO POSE' ? 'NO POSE' : 'UNKNOWN';
    const klass = joint.state === 'VISIBLE' ? 'joint-ok' : 'joint-low';
    return `<div class="joint-item"><span class="joint-name">${joint.name}</span><span class="${klass}">${label}</span></div>`;
  }).join('');
  ui.framingHint.textContent = !rawPoseSeen
    ? 'Show yourself from head to feet for Squat or OHP. This is a visibility guide only.'
    : data.basicVisibilityReady
      ? 'Required joints appear visible. This is NOT an exercise-form or model inference result.'
      : `Framing hint: ${data.missing.slice(0, 3).join(', ')}${data.missing.length > 3 ? '…' : ''}. Move so these joints are visible.`;
  const perTask = taskReadiness(state.exercise, state.landmarks);
  const n = perTask.filter((t) => t.basicVisibilityReady).length;
  ui.videoStatus.textContent = rawPoseSeen
    ? `${n}/${perTask.length} task visibility checks ready · no RF inference`
    : 'No pose detected';
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
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const points = state.landmarks;
  if (!Array.isArray(points)) return;
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
}

function loop(token, timestamp) {
  if (!state.running || token !== state.token) return;
  const v = ui.video;
  try {
    if (v.readyState >= 2 && v.currentTime !== state.lastVideoTime) {
      state.lastVideoTime = v.currentTime;
      const result = state.landmarker.detectForVideo(v, timestamp);
      state.landmarks = result.landmarks?.[0] ?? null;
      state.frames++;
      if (state.landmarks) state.poseFrames++;
      if (state.lastFrameClock) {
        const delta = timestamp - state.lastFrameClock;
        if (delta > 0) state.fps = state.fps ? state.fps * .85 + (1000 / delta) * .15 : (1000 / delta);
      }
      state.lastFrameClock = timestamp;
      drawFrame();
      if (timestamp - state.lastRendered > 220) {
        state.lastRendered = timestamp;
        updateInspector();
      }
    }
  } catch (error) {
    stopCamera();
    setStatus('POSE ENGINE ERROR', `The local pose engine stopped: ${error.message || 'unknown error'}`);
    return;
  }
  state.raf = requestAnimationFrame((time) => loop(token, time));
}

function stopCamera() {
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
    tempStream = await navigator.mediaDevices.getUserMedia({
      audio: false, video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
    });
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
    tempStream = null;
    tempLandmarker = null;
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
    setStatus('LOCAL POSE ACTIVE', 'Camera frames are processed locally; six RF models are not hosted.');
    state.raf = requestAnimationFrame((timestamp) => loop(token, timestamp));
  } catch (error) {
    tempStream?.getTracks().forEach((track) => track.stop());
    tempLandmarker?.close();
    stopCamera();
    setStatus('COULD NOT START CAMERA',
      `Check camera permission, browser support and internet for first-time model download. ${error.message || ''}`);
  }
}

ui.button.addEventListener('click', () => state.running ? stopCamera() : startCamera());
ui.exercise.addEventListener('change', () => {
  if (!TASKS[ui.exercise.value]) return;
  state.exercise = ui.exercise.value;
  ui.help.textContent = TASKS[state.exercise].subtitle;
  renderModels();
  updateInspector();
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && state.running) stopCamera();
});
window.addEventListener('pagehide', stopCamera);
ui.help.textContent = TASKS[state.exercise].subtitle;
renderModels();
updateInspector();
