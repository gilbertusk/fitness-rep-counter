import { createGenericCounter, updateGenericCounter } from '../counting/genericCounter.js';
import { createHoldTimer, updateHoldTimer, finishHold } from '../counting/holdTimer.js';
import { createClassifier, updateClassifier, resetClassifier } from '../classify/classifier.js';
import { CATALOG, exerciseName } from '../exercises.js';
import { createWindowStream, pushWindowStream } from '../features/features.js';
import { filterPose } from '../geometry/oneEuroFilter.js';
import { createDebounce, updateDebounce } from '../form/debounce.js';
import {
  evaluateForm, brokenRules, inPosition, createRepCheck, updateRepCheck, assessPoseQuality, countVisible,
} from '../form/measure.js';
import { rulesFor } from '../form/rules/index.js';
import {
  PHASE, createSession, startSession, stopSession, overrideExercise, onPrediction, onFrame, clearHistory,
} from './session.js';
import { createSpeechQueue, say, nextUtterance, setSpeechEnabled } from './speechQueue.js';

/**
 * Everything the app decides per frame, in one pure function, so main.js only wires browser APIs to
 * it and the whole loop can be tested in Node. Pure and immutable.
 *
 *   raw pose ──▶ rep counter, classifier windows          (raw: they must match training/evaluation)
 *            └▶ One Euro ──▶ overlay, form rules, plank   (smoothed: what the user sees and is told)
 */

const REP_WARNING_MS = 2500; // how long a once-per-rep warning (e.g. lockout) stays on screen
const FPS_SMOOTHING = 0.1;

/**
 * @param {{ classifierLabels?: string[]|null, classifierThresholds?: object, speech?: boolean }} options
 *   classifierLabels: the labels of a loaded exercise classifier, or null when none is available —
 *   then exercises can only be chosen by hand and reps are still counted
 */
export function createWorkout({ classifierLabels = null, classifierThresholds = {}, speech = true } = {}) {
  return Object.freeze({
    session: createSession(),
    classifier: classifierLabels ? createClassifier({ labels: classifierLabels, thresholds: classifierThresholds }) : null,
    stream: createWindowStream(),
    ...freshSet(),
    setId: 0,
    poseFilter: null,
    speech: createSpeechQueue({ enabled: speech }),
    lastTimeMs: null,
    fps: null,
  });
}

/** Per-set state, rebuilt whenever the session opens a new set. */
const freshSet = () => ({
  counter: createGenericCounter(),
  debounce: createDebounce(),
  repCheck: createRepCheck(),
  hold: createHoldTimer(),
  visible: Object.freeze([]),
  repWarning: null,
});

export const startWorkout = (state, timeMs) => Object.freeze({ ...state, session: startSession(state.session, timeMs) });

export function stopWorkout(state, timeMs) {
  return Object.freeze({ ...state, session: stopSession(state.session, timeMs), hold: finishHold(state.hold) });
}

/** Choose the exercise by hand (a catalog label), or hand control back to auto-detect (null). */
export function chooseExercise(state, label, timeMs) {
  return Object.freeze({ ...state, session: overrideExercise(state.session, label, timeMs) });
}

export const setSpeech = (state, enabled) => Object.freeze({ ...state, speech: setSpeechEnabled(state.speech, enabled) });
export const forgetHistory = (state) => Object.freeze({ ...state, session: clearHistory(state.session) });

/**
 * Fold in the classifier's scores for one window (adapters/onnxClassifier.js → classify()).
 * @param {{ poorPose?: boolean }} context poorPose: the window had too many missing frames
 */
export function applyScores(state, scores, context = {}) {
  if (!state.classifier) return state;
  const classifier = updateClassifier(state.classifier, scores, context);
  return Object.freeze({ ...state, classifier, session: onPrediction(state.session, classifier.prediction) });
}

const currentLabel = (session) => session.set?.label ?? session.manual ?? null;

function formStep(state, smoothed, ruleSet, timeMs, size, repCompleted) {
  const evaluation = evaluateForm(smoothed, ruleSet, size);
  const { state: debounce, visible } = updateDebounce(state.debounce, brokenRules(evaluation), timeMs);
  const { state: repCheck, failed } = updateRepCheck(state.repCheck, evaluation, ruleSet, repCompleted);
  const repWarning = failed.length ? { ...failed[0], untilMs: timeMs + REP_WARNING_MS }
    : state.repWarning && state.repWarning.untilMs > timeMs ? state.repWarning : null;
  const messages = (ruleSet?.rules ?? []).filter((r) => visible.includes(r.id)).map((r) => r.message);
  return { evaluation, debounce, repCheck, repWarning, visible: Object.freeze(visible), failed,
    warnings: [...messages, ...(repWarning ? [repWarning.message] : [])],
    // Whether any rule could actually measure this frame: silence is not the same as good form.
    checked: evaluation.results.some((r) => r.value !== null && Number.isFinite(r.value)) };
}

function speechStep(state, form, reps, previousReps, timeMs) {
  let speech = state.speech;
  const newWarning = form.visible.find((id) => !state.visible.includes(id));
  if (reps > previousReps) speech = say(speech, String(reps));
  if (newWarning) speech = say(speech, form.warnings[form.visible.indexOf(newWarning)]);
  if (form.failed.length) speech = say(speech, form.failed[0].message);
  return nextUtterance(speech, timeMs);
}

/**
 * Fold in one video frame.
 * @param {{ frame: Array|null, timeMs: number, size: { width: number, height: number } }} input
 *   frame: MediaPipe's 33 raw landmarks, or null when no pose was found
 * @returns {{ state: object, view: object, effects: { speak: string|null, classify: Array, closedSets: Array } }}
 *   classify: windows to send to the classifier; feed its scores back with applyScores()
 */
export function stepFrame(state, { frame, timeMs, size }) {
  const fresh = state.session.setId !== state.setId ? { ...freshSet(), setId: state.session.setId,
    classifier: state.classifier && resetClassifier(state.classifier) } : {};
  const base = { ...state, ...fresh };

  const { state: poseFilter, landmarks: smoothed } = filterPose(base.poseFilter, frame, timeMs);
  const counter = updateGenericCounter(base.counter, frame, timeMs, size);
  const { state: stream, windows } = pushWindowStream(base.stream, frame, timeMs, size);
  const label = currentLabel(base.session);
  const ruleSet = rulesFor(label);
  const form = formStep(base, smoothed, ruleSet, timeMs, size, counter.count > base.counter.count);
  const plank = CATALOG[label]?.task === 'hold';
  const hold = plank ? updateHoldTimer(base.hold, inPosition(form.evaluation), timeMs) : base.hold;

  const session = onFrame(base.session, { timeMs, moving: counter.moving, reps: counter.count,
    holding: hold.holding, holdMs: hold.bestMs, warning: form.visible.length > 0 });
  const { state: speech, text } = speechStep(base, form, counter.count, base.counter.count, timeMs);
  const interval = base.lastTimeMs === null ? null : timeMs - base.lastTimeMs;
  const fps = interval > 0 ? (base.fps === null ? 1000 / interval : base.fps + FPS_SMOOTHING * (1000 / interval - base.fps)) : base.fps;

  const next = Object.freeze({ ...base, poseFilter, counter, stream, debounce: form.debounce, repCheck: form.repCheck,
    repWarning: form.repWarning, visible: form.visible, hold, session, speech, lastTimeMs: timeMs, fps });
  return {
    state: next,
    view: viewOf(next, { smoothed, label, ruleSet, warnings: form.warnings, formChecked: form.checked, size, raw: frame, timeMs }),
    effects: { speak: text, classify: windows, closedSets: session.sets.slice(base.session.sets.length) },
  };
}

/** The set that just ended, while resting after it: what the "set selesai" banner and rest timer show. */
function restOf(session, timeMs) {
  const last = session.sets[session.sets.length - 1];
  if (session.phase !== PHASE.RESTING || !last) return null;
  return Object.freeze({ sinceMs: timeMs - last.endMs, label: last.label, name: last.label ? exerciseName(last.label) : null,
    reps: last.reps, holdMs: last.holdMs });
}

function viewOf(state, { smoothed, label, ruleSet, warnings, formChecked, size, raw, timeMs }) {
  const entry = CATALOG[label] ?? null;
  const { set } = state.session;
  return Object.freeze({
    phase: state.session.phase,
    manual: state.session.manual,
    autoAvailable: state.classifier !== null,
    label,
    name: label ? exerciseName(label) : null,
    task: entry?.task ?? 'reps',
    status: entry?.status ?? null,
    note: entry?.note ?? null,
    confidence: state.session.phase === PHASE.COUNTING && !state.session.manual ? state.session.set?.confidence ?? null : null,
    candidate: state.session.phase === PHASE.DETECTING ? state.session.candidate : null,
    reps: set?.reps ?? 0,
    repsWithWarning: set?.repsWithWarning ?? 0,
    setDurationMs: set ? timeMs - set.startMs : null,
    rest: restOf(state.session, timeMs),
    visibleKeypoints: countVisible(raw),
    hold: { holding: state.hold.holding, currentMs: state.hold.currentMs, bestMs: state.hold.bestMs },
    warnings: Object.freeze(warnings),
    hasRules: ruleSet !== null,
    formChecked,
    quality: assessPoseQuality(raw, size, { points: ruleSet?.sidePoints, view: ruleSet?.view }),
    fps: state.fps,
    landmarks: smoothed,
    sets: state.session.sets,
  });
}
