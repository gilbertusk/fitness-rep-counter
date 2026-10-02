import { createPoseLandmarker } from '../../../app/src/adapters/poseLandmarker.js';
import { createExerciseClassifier } from '../../../app/src/adapters/onnxClassifier.js';
import { createPoseRenderer } from '../../../app/src/ui/overlay.js';
import { createWorkout, startWorkout, chooseExercise, applyScores, stepFrame } from '../../../app/src/core/session/workout.js';
import { createWindowStream, pushWindowStream } from '../../../app/src/core/features/features.js';
import { createGenericCounter, updateGenericCounter } from '../../../app/src/core/counting/genericCounter.js';
import { summarize, framesPerSecond, groupAssets, toMarkdown } from './stats.js';

/**
 * Measures the real app pipeline in this browser, in two passes over one video:
 *
 *  1. live — what the app does per frame, at the video's own speed: MediaPipe pose → workout.stepFrame
 *     → overlay drawing, plus the classifier on every window when one is loaded. Gives end-to-end FPS
 *     and the pose and classifier latency.
 *  2. replay — the recorded landmarks go again through features, counter and the whole stepFrame.
 *     These take well under a millisecond, below the 0.1 ms resolution of performance.now() without
 *     cross-origin isolation, so each frame's call is repeated REPEAT_CORE times on the same input and
 *     averaged. That is legitimate because core functions are pure: the same state in, the same out.
 *
 * URL options: ?repeat=3 (video loops) · ?exercise=push_up · ?label=<device name>
 *              · ?model=<.onnx url>&labels=<labels.json url> (defaults: app/models/)
 */

const REPEAT_CORE = 20;
const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const now = () => performance.now();

const fixture = (h264) => `../../app/tests/fixtures/videos/push-up_17.${h264 ? 'mp4' : 'webm'}`;

async function loadClassifier() {
  const modelUrl = params.get('model') ?? '../../app/models/exercise_classifier.onnx';
  const labelsUrl = params.get('labels') ?? '../../app/models/labels.json';
  try {
    const classifier = await createExerciseClassifier({ modelUrl, labelsUrl });
    const note = (await (await fetch(labelsUrl)).json()).note ?? null;
    return { classifier, note };
  } catch (error) {
    return { classifier: null, note: `Pengenal latihan tidak diukur: ${error.message}.` };
  }
}

function playOnce(video) {
  return new Promise((resolve) => {
    video.currentTime = 0;
    video.addEventListener('ended', resolve, { once: true });
    video.play();
  });
}

/** Pass 1: the app's own loop. Returns the recorded frames and the live timings. */
async function livePass(video, pose, drawPose, classifier, repeat) {
  const exercise = params.get('exercise') ?? 'push_up';
  let workout = createWorkout({ classifierLabels: classifier?.labels ?? null, classifierThresholds: classifier?.thresholds, speech: false });
  workout = chooseExercise(startWorkout(workout, now()), exercise, now());
  const record = { frames: [], frameTimes: [], poseMs: [], frameMs: [], classifyMs: [], firstPoseMs: null };
  let busy = false;
  let lastVideoTime = -1;

  const classify = async (windows) => {
    if (!classifier || busy || !windows.length) return;
    busy = true;
    for (const { window } of windows) {
      const start = now();
      const scores = await classifier.classify(window);
      record.classifyMs.push(now() - start);
      workout = applyScores(workout, scores);
    }
    busy = false;
  };

  const onFrame = () => {
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    const size = { width: video.videoWidth, height: video.videoHeight };
    const start = now();
    const frame = pose.detect(video, start) ?? null;
    const afterPose = now();
    const { state, view, effects } = stepFrame(workout, { frame, timeMs: start, size });
    workout = state;
    drawPose(view.landmarks);
    record.frameMs.push(now() - start);
    record.poseMs.push(afterPose - start);
    record.frameTimes.push(start);
    record.frames.push({ frame, timeMs: start, size });
    if (frame && record.firstPoseMs === null) record.firstPoseMs = afterPose;
    classify(effects.classify);
  };

  // Same scheduling as app/src/main.js (requestAnimationFrame + "is there a new video frame?"), not
  // requestVideoFrameCallback: when pose takes longer than one video frame, the two skip differently.
  let running = true;
  const schedule = () => requestAnimationFrame(() => { onFrame(); if (running) schedule(); });
  schedule();
  for (let i = 0; i < repeat; i += 1) await playOnce(video);
  running = false;
  while (busy) await new Promise((resolve) => setTimeout(resolve, 20));
  return { record, workout };
}

/** Mean time of `REPEAT_CORE` calls of a pure step on the same input, per recorded frame. */
function timePure(frames, initial, step) {
  let state = initial;
  return frames.map((input) => {
    const start = now();
    let next = state;
    for (let i = 0; i < REPEAT_CORE; i += 1) next = step(state, input);
    const elapsed = (now() - start) / REPEAT_CORE;
    state = next;
    return elapsed;
  });
}

/** Pass 2: core stages replayed on the recorded landmarks. */
function replayPass(frames, classifier) {
  const exercise = params.get('exercise') ?? 'push_up';
  const t0 = frames[0]?.timeMs ?? 0;
  const workout = chooseExercise(startWorkout(createWorkout({ classifierLabels: classifier?.labels ?? null, speech: false }), t0), exercise, t0);
  return {
    features: timePure(frames, createWindowStream(), (s, f) => pushWindowStream(s, f.frame, f.timeMs, f.size).state),
    counter: timePure(frames, createGenericCounter(), (s, f) => updateGenericCounter(s, f.frame, f.timeMs, f.size)),
    step: timePure(frames, workout, (s, f) => stepFrame(s, f).state),
  };
}

async function runBenchmark(video, source) {
  const status = $('status');
  status.textContent = 'Memuat model pose…';
  const pose = await createPoseLandmarker();
  const poseReadyMs = now();
  status.textContent = 'Memuat pengenal latihan (bila ada)…';
  const { classifier, note } = await loadClassifier();
  const classifierReadyMs = classifier ? now() : null;

  video.src = source.url;
  await new Promise((resolve) => video.addEventListener('loadeddata', resolve, { once: true }));
  const canvas = $('overlay');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const repeat = Math.max(1, Number(params.get('repeat') ?? 3));

  status.textContent = `Mengukur: ${repeat} × putar video…`;
  const { record, workout } = await livePass(video, pose, createPoseRenderer(canvas, pose.vision), classifier, repeat);
  status.textContent = 'Mengulang landmark lewat core…';
  const replay = replayPass(record.frames, classifier);

  const result = {
    dateMs: Date.now(),
    device: { label: params.get('label') ?? $('label').value, userAgent: navigator.userAgent,
      cores: navigator.hardwareConcurrency ?? null, delegate: pose.delegate, renderer: pose.renderer },
    video: { name: source.name, width: video.videoWidth, height: video.videoHeight, durationS: video.duration, repeat,
      fps: source.fps },
    load: { poseReadyMs, firstPoseMs: record.firstPoseMs, classifierReadyMs },
    live: { fps: framesPerSecond(record.frameTimes), frames: record.frames.length,
      posesFound: record.frames.filter((f) => f.frame).length, reps: workout.session.set?.reps ?? 0 },
    stages: [
      { label: 'Pose (MediaPipe detectForVideo)', summary: summarize(record.poseMs) },
      { label: 'Fitur (pushWindowStream)', summary: summarize(replay.features) },
      { label: 'Penghitung (updateGenericCounter)', summary: summarize(replay.counter) },
      { label: 'Core total (stepFrame: fitur + counter + form + sesi)', summary: summarize(replay.step) },
      { label: 'Pengenal (ONNX, per window)', summary: summarize(record.classifyMs) },
      { label: 'Frame utuh (pose + stepFrame + overlay)', summary: summarize(record.frameMs) },
    ],
    assets: groupAssets(performance.getEntriesByType('resource')),
    notes: [note, `Repetisi terhitung sepanjang benchmark (video diputar ${repeat}×): ${workout.session.set?.reps ?? 0} — angka ini bukan evaluasi akurasi.`].filter(Boolean),
  };
  status.textContent = 'Selesai.';
  return result;
}

function show(result) {
  const markdown = toMarkdown(result);
  $('report').textContent = markdown;
  $('btn-copy').disabled = false;
  $('btn-copy').onclick = () => navigator.clipboard.writeText(markdown);
  window.__benchResult = { result, markdown };
}

async function start(source) {
  $('btn-run').disabled = true;
  try {
    show(await runBenchmark($('video'), source));
  } catch (error) {
    $('status').textContent = `Gagal: ${error.message}`;
    window.__benchResult = { error: error.message };
  } finally {
    $('btn-run').disabled = false;
  }
}

const h264 = document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"') !== '';
let source = { url: fixture(h264), name: fixture(h264).split('/').pop(), fps: 30 };
$('label').value = params.get('label') ?? '';
$('video-file').addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (file) source = { url: URL.createObjectURL(file), name: file.name, fps: null }; // stays on this device
});
$('btn-run').addEventListener('click', () => start(source));
if (params.has('autostart')) start(source);
