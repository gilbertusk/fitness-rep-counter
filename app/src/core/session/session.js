/**
 * Workout session state machine. Pure and immutable.
 *
 *   IDLE ──start──▶ DETECTING ──(same confident label for 2 windows)──▶ COUNTING
 *                       ▲                                                   │
 *                       └──────── movement again ◀── RESTING ◀──(still > 3 s)┘
 *
 * A set begins when movement begins, not when the label locks, so reps made while the exercise is
 * still being recognised count towards the set. The label stays locked while COUNTING. A manual
 * choice (override) switches auto-detect off: sets then start straight in COUNTING with that label.
 * Plank is a hold, so "still" alone does not end a plank set — the hold ending does.
 */

export const PHASE = Object.freeze({ IDLE: 'idle', DETECTING: 'detecting', COUNTING: 'counting', RESTING: 'resting' });

export const SESSION_DEFAULTS = Object.freeze({
  stableWindows: 2,  // consecutive confident windows with one label before it locks
  restMs: 3000,      // no movement for this long ends the set
});

export function createSession(options = {}) {
  return Object.freeze({
    config: { ...SESSION_DEFAULTS, ...options },
    phase: PHASE.IDLE,
    manual: null,
    candidate: null,
    streak: 0,
    setId: 0,
    set: null,
    lastActiveMs: null,
    sets: Object.freeze([]),
  });
}

const newSet = (label, timeMs, confidence = null) => Object.freeze({
  label, confidence, startMs: timeMs, reps: 0, repsWithWarning: 0, warnedThisRep: false, holdMs: 0,
});

/** Close the current set; it is kept in history only if something happened in it. */
function closeSet(state, timeMs) {
  const { set } = state;
  if (!set || (set.reps === 0 && set.holdMs === 0)) return { ...state, set: null };
  const record = Object.freeze({
    label: set.label, manual: state.manual !== null, reps: set.reps, repsWithWarning: set.repsWithWarning,
    holdMs: set.holdMs, startMs: set.startMs, endMs: timeMs, durationMs: timeMs - set.startMs,
  });
  return { ...state, set: null, sets: Object.freeze([...state.sets, record]) };
}

/** Begin a new set: COUNTING with the manual label, or DETECTING when auto-detect is on. */
function openSet(state, timeMs) {
  return {
    ...state,
    phase: state.manual ? PHASE.COUNTING : PHASE.DETECTING,
    set: newSet(state.manual, timeMs),
    setId: state.setId + 1,
    candidate: null,
    streak: 0,
    lastActiveMs: timeMs,
  };
}

export function startSession(state, timeMs) {
  if (state.phase !== PHASE.IDLE) return state;
  return Object.freeze(openSet(state, timeMs));
}

export function stopSession(state, timeMs) {
  return Object.freeze({ ...closeSet(state, timeMs), phase: PHASE.IDLE, candidate: null, streak: 0 });
}

/** Choose an exercise by hand (label), or hand control back to auto-detect (null). */
export function overrideExercise(state, label, timeMs) {
  const closed = { ...closeSet(state, timeMs), manual: label ?? null };
  if (state.phase === PHASE.IDLE) return Object.freeze(closed);
  return Object.freeze(openSet(closed, timeMs));
}

/**
 * Fold in one classifier window (core/classify/classifier.js `prediction`). Only DETECTING listens,
 * and only while auto-detect is on.
 * @param {{ label: string, confidence: number, uncertain: boolean }} prediction
 */
export function onPrediction(state, prediction) {
  if (state.phase !== PHASE.DETECTING || state.manual || !prediction) return state;
  if (prediction.uncertain) return Object.freeze({ ...state, candidate: null, streak: 0 });
  const streak = prediction.label === state.candidate ? state.streak + 1 : 1;
  if (streak < state.config.stableWindows) {
    return Object.freeze({ ...state, candidate: prediction.label, streak });
  }
  return Object.freeze({ ...state, phase: PHASE.COUNTING, candidate: prediction.label, streak,
    set: Object.freeze({ ...state.set, label: prediction.label, confidence: prediction.confidence }) });
}

function trackReps(set, reps, warning) {
  const warnedThisRep = set.warnedThisRep || warning;
  if (reps <= set.reps) return Object.freeze({ ...set, warnedThisRep });
  return Object.freeze({ ...set, reps, warnedThisRep: false,
    repsWithWarning: set.repsWithWarning + (warnedThisRep ? 1 : 0) });
}

/**
 * Fold in one frame.
 * @param {{ timeMs: number, moving: boolean, reps?: number, holdMs?: number, holding?: boolean, warning?: boolean }} frame
 *   reps: the rep counter's count since this set began; holdMs/holding: the plank timer; warning:
 *   whether a form warning is showing, so reps made with one can be counted
 */
export function onFrame(state, { timeMs, moving, reps = 0, holdMs = 0, holding = false, warning = false }) {
  if (state.phase === PHASE.IDLE) return state;
  const active = moving || holding;
  if (state.phase === PHASE.RESTING) {
    return active ? Object.freeze(openSet(state, timeMs)) : state;
  }
  const lastActiveMs = active ? timeMs : state.lastActiveMs;
  const set = Object.freeze({ ...trackReps(state.set, reps, warning), holdMs: Math.max(state.set.holdMs, holdMs) });
  const next = { ...state, set, lastActiveMs };
  if (timeMs - lastActiveMs <= state.config.restMs) return Object.freeze(next);
  return Object.freeze({ ...closeSet(next, timeMs), phase: PHASE.RESTING, candidate: null, streak: 0 });
}

/** Forget the finished sets (the "hapus riwayat" button). */
export function clearHistory(state) {
  return Object.freeze({ ...state, sets: Object.freeze([]) });
}
