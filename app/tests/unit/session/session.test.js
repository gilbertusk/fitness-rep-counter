import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  PHASE, SESSION_DEFAULTS, createSession, startSession, stopSession, overrideExercise, onPrediction, onFrame, clearHistory,
} from '../../../src/core/session/session.js';

const confident = (label, confidence = 0.9) => ({ label, confidence, uncertain: false });
const unsure = (label) => ({ label, confidence: 0.4, uncertain: true });

/** Frames every 100 ms from `from` to `to`, with the given flags. */
function frames(state, from, to, flags = {}) {
  let s = state;
  for (let t = from; t <= to; t += 100) s = onFrame(s, { timeMs: t, ...flags });
  return s;
}

const started = () => startSession(createSession(), 0);

test('starting enters DETECTING with a fresh set', () => {
  const state = started();
  assert.equal(state.phase, PHASE.DETECTING);
  assert.equal(state.setId, 1);
  assert.equal(state.set.label, null);
  assert.equal(startSession(state, 50), state, 'starting twice changes nothing');
});

test('two confident windows with one label lock it and start COUNTING', () => {
  assert.equal(SESSION_DEFAULTS.stableWindows, 2);
  const once = onPrediction(started(), confident('squat'));
  assert.equal(once.phase, PHASE.DETECTING);
  const twice = onPrediction(once, confident('squat', 0.8));
  assert.equal(twice.phase, PHASE.COUNTING);
  assert.equal(twice.set.label, 'squat');
  assert.equal(twice.set.confidence, 0.8);
});

test('a change of mind or an uncertain window restarts the streak', () => {
  let state = onPrediction(started(), confident('squat'));
  state = onPrediction(state, confident('lunge'));
  assert.equal(state.phase, PHASE.DETECTING);
  assert.equal(state.candidate, 'lunge');
  state = onPrediction(state, unsure('lunge'));
  assert.equal(state.streak, 0);
  assert.equal(onPrediction(state, confident('lunge')).phase, PHASE.DETECTING);
});

test('the label stays locked while COUNTING', () => {
  let state = onPrediction(onPrediction(started(), confident('squat')), confident('squat'));
  state = onPrediction(onPrediction(state, confident('lunge')), confident('lunge'));
  assert.equal(state.set.label, 'squat');
});

test('reps made while still recognising the exercise count towards the set', () => {
  let state = frames(started(), 0, 2000, { moving: true, reps: 2 });
  state = onPrediction(onPrediction(state, confident('squat')), confident('squat'));
  state = frames(state, 2100, 4000, { moving: true, reps: 5 });
  assert.equal(state.set.reps, 5);
});

test('more than three seconds without movement closes the set and rests', () => {
  assert.equal(SESSION_DEFAULTS.restMs, 3000);
  let state = onPrediction(onPrediction(started(), confident('squat')), confident('squat'));
  state = frames(state, 0, 4000, { moving: true, reps: 6 });
  state = frames(state, 4100, 7000, { moving: false, reps: 6 });
  assert.equal(state.phase, PHASE.COUNTING, 'exactly three seconds is not yet rest');
  state = frames(state, 7100, 7200, { moving: false, reps: 6 });
  assert.equal(state.phase, PHASE.RESTING);
  assert.equal(state.set, null);
  assert.equal(state.sets.length, 1);
  assert.deepEqual({ ...state.sets[0] }, {
    label: 'squat', manual: false, reps: 6, repsWithWarning: 0, holdMs: 0, startMs: 0, endMs: 7100, durationMs: 7100,
  });
});

test('moving again after a rest starts a new set and detects afresh', () => {
  let state = onPrediction(onPrediction(started(), confident('squat')), confident('squat'));
  state = frames(state, 0, 2000, { moving: true, reps: 3 });
  state = frames(state, 2100, 5500, { moving: false, reps: 3 });
  assert.equal(state.phase, PHASE.RESTING);
  state = onFrame(state, { timeMs: 6000, moving: true });
  assert.equal(state.phase, PHASE.DETECTING);
  assert.equal(state.setId, 2);
  assert.equal(state.set.label, null);
});

test('a stretch of stillness with no reps leaves no empty set in the history', () => {
  const state = frames(started(), 0, 4000, { moving: false });
  assert.equal(state.phase, PHASE.RESTING);
  assert.deepEqual(state.sets, []);
});

test('reps made while a warning was showing are counted as such', () => {
  let state = onPrediction(onPrediction(started(), confident('squat')), confident('squat'));
  state = onFrame(state, { timeMs: 100, moving: true, reps: 0, warning: true });
  state = onFrame(state, { timeMs: 200, moving: true, reps: 1 });     // rep 1: warned during it
  state = onFrame(state, { timeMs: 300, moving: true, reps: 2 });     // rep 2: clean
  state = onFrame(state, { timeMs: 400, moving: true, reps: 2, warning: true });
  state = onFrame(state, { timeMs: 500, moving: true, reps: 3 });     // rep 3: warned
  assert.equal(state.set.reps, 3);
  assert.equal(state.set.repsWithWarning, 2);
});

test('a manual choice switches auto-detect off and counts straight away', () => {
  let state = overrideExercise(started(), 'push_up', 1000);
  assert.equal(state.phase, PHASE.COUNTING);
  assert.equal(state.set.label, 'push_up');
  assert.equal(state.manual, 'push_up');
  state = onPrediction(onPrediction(state, confident('squat')), confident('squat'));
  assert.equal(state.set.label, 'push_up', 'predictions are ignored in manual mode');
});

test('switching exercise by hand closes the running set', () => {
  let state = overrideExercise(started(), 'push_up', 0);
  state = frames(state, 0, 1000, { moving: true, reps: 4 });
  state = overrideExercise(state, 'squat', 1100);
  assert.equal(state.sets.length, 1);
  assert.equal(state.sets[0].label, 'push_up');
  assert.equal(state.sets[0].manual, true);
  assert.equal(state.set.label, 'squat');
});

test('handing control back to auto-detect returns to DETECTING', () => {
  const state = overrideExercise(overrideExercise(started(), 'push_up', 0), null, 500);
  assert.equal(state.manual, null);
  assert.equal(state.phase, PHASE.DETECTING);
});

test('a manual choice before starting is remembered for the first set', () => {
  const state = startSession(overrideExercise(createSession(), 'plank', 0), 100);
  assert.equal(state.phase, PHASE.COUNTING);
  assert.equal(state.set.label, 'plank');
});

test('in manual mode, moving after a rest starts the next set of the same exercise', () => {
  let state = overrideExercise(started(), 'squat', 0);
  state = frames(state, 0, 1000, { moving: true, reps: 2 });
  state = frames(state, 1100, 4500, { moving: false, reps: 2 });
  state = onFrame(state, { timeMs: 5000, moving: true });
  assert.equal(state.phase, PHASE.COUNTING);
  assert.equal(state.set.label, 'squat');
});

test('a plank set lasts as long as the hold, not as long as there is movement', () => {
  let state = overrideExercise(started(), 'plank', 0);
  state = frames(state, 0, 20000, { moving: false, holding: true, holdMs: 20000 });
  assert.equal(state.phase, PHASE.COUNTING, 'a still plank is not rest');
  state = frames(state, 20100, 23500, { moving: false, holding: false, holdMs: 20000 });
  assert.equal(state.phase, PHASE.RESTING);
  assert.equal(state.sets[0].holdMs, 20000);
});

test('stopping closes the set and returns to IDLE, where frames are ignored', () => {
  let state = overrideExercise(started(), 'squat', 0);
  state = frames(state, 0, 1000, { moving: true, reps: 3 });
  state = stopSession(state, 1100);
  assert.equal(state.phase, PHASE.IDLE);
  assert.equal(state.sets.length, 1);
  assert.equal(onFrame(state, { timeMs: 2000, moving: true }), state);
});

test('clearing the history keeps the session running', () => {
  let state = overrideExercise(started(), 'squat', 0);
  state = frames(state, 0, 1000, { moving: true, reps: 3 });
  state = clearHistory(stopSession(state, 1100));
  assert.deepEqual(state.sets, []);
});

test('updates never mutate the previous state', () => {
  const before = onPrediction(started(), confident('squat'));
  const snapshot = JSON.stringify(before);
  onPrediction(before, confident('squat'));
  onFrame(before, { timeMs: 9000, moving: false });
  overrideExercise(before, 'plank', 50);
  assert.equal(JSON.stringify(before), snapshot);
  assert.ok(Object.isFrozen(before));
});
