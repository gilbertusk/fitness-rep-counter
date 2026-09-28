import { MEDIAPIPE_BUNDLE_URL, MEDIAPIPE_WASM_URL, POSE_MODEL_URL } from './config.js';
import { EXERCISES } from './core/exercises.js';
import { measureExercise } from './core/pose.js';
import { createCounter, updateCounter, FEEDBACK } from './core/repCounter.js';

const $ = (id) => document.getElementById(id);
const ui = {
  exercise: $('exercise'),
  cameraButton: $('btn-camera'),
  fileInput: $('video-file'),
  resetButton: $('btn-reset'),
  stage: document.querySelector('.stage'),
  video: $('video'),
  canvas: $('overlay'),
  status: $('status'),
  count: $('count'),
  angle: $('angle'),
  feedback: $('feedback'),
  warnings: $('warnings'),
};

let vision = null;
let landmarker = null;
let drawing = null;
let counter = createCounter();
let lastVideoTime = -1;

const currentExercise = () => EXERCISES[ui.exercise.value];

function setStatus(message) {
  ui.status.textContent = message;
}

async function loadModel() {
  vision = await import(MEDIAPIPE_BUNDLE_URL);
  const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
  landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
  drawing = new vision.DrawingUtils(ui.canvas.getContext('2d'));
}

function resetCounter() {
  counter = createCounter();
  renderStats({ angle: null, warnings: [] });
}

function renderStats({ angle, warnings }) {
  ui.count.textContent = counter.count;
  ui.angle.textContent = angle === null ? '–' : `${Math.round(angle)}°`;
  ui.feedback.textContent = counter.feedback ?? '–';
  ui.feedback.className = `feedback ${counter.feedback === FEEDBACK.GOOD ? 'good' : counter.feedback ? 'bad' : ''}`;
  ui.warnings.replaceChildren(...warnings.map((text) => Object.assign(document.createElement('li'), { textContent: text })));
}

function drawPose(pose) {
  const ctx = ui.canvas.getContext('2d');
  ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  if (!pose) return;

  drawing.drawConnectors(pose, vision.PoseLandmarker.POSE_CONNECTIONS, { color: '#60a5fa', lineWidth: 4 });
  drawing.drawLandmarks(pose, { color: '#f97316', radius: 4 });
}

function processFrame() {
  const { video, canvas } = ui;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

    const result = landmarker.detectForVideo(video, performance.now());
    const pose = result.landmarks[0];
    const measurement = measureExercise(pose, currentExercise(), { width: canvas.width, height: canvas.height });

    counter = updateCounter(counter, measurement.angle, currentExercise());
    drawPose(pose);
    renderStats(measurement);
    setStatus(pose ? '' : 'Tubuh tidak terdeteksi, mundur sedikit dari kamera');
  }
  requestAnimationFrame(processFrame);
}

function stopCurrentSource() {
  const stream = ui.video.srcObject;
  stream?.getTracks().forEach((track) => track.stop());
  ui.video.srcObject = null;
  if (ui.video.src) URL.revokeObjectURL(ui.video.src);
  ui.video.removeAttribute('src');
}

async function startCamera() {
  try {
    stopCurrentSource();
    ui.video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    ui.stage.classList.add('mirrored');
    await ui.video.play();
    resetCounter();
  } catch (error) {
    console.error(error);
    setStatus('Kamera tidak bisa diakses. Cek izin kamera di browser.');
  }
}

async function startVideoFile(file) {
  if (!file) return;
  stopCurrentSource();
  ui.video.src = URL.createObjectURL(file);
  ui.video.loop = true;
  ui.stage.classList.remove('mirrored');
  await ui.video.play();
  resetCounter();
}

async function init() {
  ui.resetButton.addEventListener('click', resetCounter);
  ui.exercise.addEventListener('change', resetCounter);
  ui.cameraButton.addEventListener('click', startCamera);
  ui.fileInput.addEventListener('change', (event) => startVideoFile(event.target.files[0]));

  try {
    await loadModel();
  } catch (error) {
    console.error(error);
    setStatus('Gagal memuat model pose. Cek koneksi internet lalu muat ulang halaman.');
    return;
  }

  ui.cameraButton.disabled = false;
  ui.fileInput.disabled = false;
  setStatus('Model siap. Nyalakan kamera atau pilih video.');
  requestAnimationFrame(processFrame);
}

init();
