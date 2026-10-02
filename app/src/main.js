import { CATALOG } from './core/exercises.js';
import { usableWindows } from './core/features/features.js';
import {
  createWorkout, startWorkout, stopWorkout, chooseExercise, setSpeech, forgetHistory, applyScores, stepFrame,
} from './core/session/workout.js';
import { createPoseLandmarker } from './adapters/poseLandmarker.js';
import { createExerciseClassifier } from './adapters/onnxClassifier.js';
import { createSpeaker } from './adapters/speech.js';
import { loadHistory, saveHistory, clearStoredHistory } from './adapters/storage.js';
import { startCamera, startVideoFile } from './ui/camera.js';
import { createPoseRenderer } from './ui/overlay.js';
import { renderPanel } from './ui/panel.js';
import { renderHistory } from './ui/history.js';

/** Wiring only: browser APIs (adapters) → the pure workout loop (core) → the page (ui). */

const $ = (id) => document.getElementById(id);
const ui = {
  exercise: $('exercise'), cameraButton: $('btn-camera'), facingButton: $('btn-facing'), fileInput: $('video-file'),
  resetButton: $('btn-reset'), speech: $('speech'), modelStatus: $('model-status'),
  stage: document.querySelector('.stage'), video: $('video'), canvas: $('overlay'), status: $('status'), fps: $('fps'),
  phase: $('phase'), exerciseName: $('exercise-name'), exerciseMeta: $('exercise-meta'),
  repsCard: $('reps-card'), count: $('count'), holdCard: $('hold-card'), hold: $('hold'), holdBest: $('hold-best'),
  warnings: $('warnings'), formNote: $('form-note'),
  history: $('history'), historyEmpty: $('history-empty'), clearButton: $('btn-clear'),
};

const speaker = createSpeaker();
let detectPose = null;
let drawPose = null;
let classifier = null;
let workout = null;
let history = loadHistory();
let facing = 'user';
let cameraOn = false;
let lastVideoTime = -1;
let classifying = false;

const now = () => performance.now();

function fillExerciseMenu() {
  for (const [label, entry] of Object.entries(CATALOG)) {
    ui.exercise.append(Object.assign(document.createElement('option'), { value: label, textContent: entry.name }));
  }
}

async function loadModels() {
  const pose = await createPoseLandmarker();
  detectPose = pose.detect;
  drawPose = createPoseRenderer(ui.canvas, pose.vision);
  try {
    classifier = await createExerciseClassifier();
  } catch {
    classifier = null; // stage 1 has not produced app/models/ yet: manual choice only
    ui.modelStatus.hidden = false;
    ui.modelStatus.textContent = 'Pengenal latihan otomatis belum tersedia (model belum dilatih). '
      + 'Pilih latihan untuk koreksi form — repetisi tetap dihitung.';
  }
  workout = createWorkout({ classifierLabels: classifier?.labels ?? null, classifierThresholds: classifier?.thresholds,
    speech: ui.speech.checked });
}

/** A new source or "Set baru": close what was running, then start counting afresh. */
function restartWorkout() {
  const t = now();
  workout = startWorkout(stopWorkout(workout, t), t);
  const choice = ui.exercise.value === 'auto' ? null : ui.exercise.value;
  if (choice) workout = chooseExercise(workout, choice, t);
}

async function classify(windows) {
  if (!classifier || classifying || !windows.length) return;
  classifying = true;
  try {
    for (const { window, missingRatio } of windows) {
      const scores = await classifier.classify(window);
      workout = applyScores(workout, scores, { poorPose: !usableWindows([missingRatio])[0] });
    }
  } finally {
    classifying = false;
  }
}

function keepHistory(closedSets) {
  if (!closedSets.length) return;
  history = [...history, ...closedSets.map((set) => ({ ...set, savedAt: Date.now() }))];
  saveHistory(history);
  renderHistory(ui.history, ui.historyEmpty, history);
}

function processFrame() {
  const { video, canvas } = ui;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

    const t = now();
    const frame = detectPose(video, t) ?? null;
    const { state, view, effects } = stepFrame(workout, { frame, timeMs: t, size: { width: video.videoWidth, height: video.videoHeight } });
    workout = state;
    drawPose(view.landmarks);
    renderPanel(ui, view);
    if (effects.speak) speaker.speak(effects.speak);
    keepHistory(effects.closedSets);
    classify(effects.classify);
  }
  requestAnimationFrame(processFrame);
}

async function useCamera() {
  try {
    const mirrored = await startCamera(ui.video, facing);
    ui.stage.classList.toggle('mirrored', mirrored);
    cameraOn = true;
    restartWorkout();
  } catch (error) {
    console.error(error);
    ui.status.textContent = 'Kamera tidak bisa diakses. Cek izin kamera di browser.';
  }
}

function bindControls() {
  ui.cameraButton.addEventListener('click', useCamera);
  ui.facingButton.addEventListener('click', () => {
    facing = facing === 'user' ? 'environment' : 'user';
    ui.facingButton.textContent = facing === 'user' ? 'Kamera belakang' : 'Kamera depan';
    if (cameraOn) useCamera();
  });
  ui.fileInput.addEventListener('change', async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    cameraOn = false;
    ui.stage.classList.remove('mirrored');
    await startVideoFile(ui.video, file);
    restartWorkout();
  });
  ui.resetButton.addEventListener('click', restartWorkout);
  ui.exercise.addEventListener('change', () => {
    workout = chooseExercise(workout, ui.exercise.value === 'auto' ? null : ui.exercise.value, now());
  });
  ui.speech.addEventListener('change', () => { workout = setSpeech(workout, ui.speech.checked); });
  ui.clearButton.addEventListener('click', () => {
    history = [];
    clearStoredHistory();
    workout = forgetHistory(workout);
    renderHistory(ui.history, ui.historyEmpty, history);
  });
}

async function init() {
  fillExerciseMenu();
  renderHistory(ui.history, ui.historyEmpty, history);
  bindControls();
  try {
    await loadModels();
  } catch (error) {
    console.error(error);
    ui.status.textContent = 'Gagal memuat model pose. Cek koneksi internet lalu muat ulang halaman.';
    return;
  }
  [ui.cameraButton, ui.facingButton, ui.fileInput].forEach((control) => { control.disabled = false; });
  ui.status.textContent = 'Model siap. Nyalakan kamera atau pilih video.';
  requestAnimationFrame(processFrame);
}

init();
