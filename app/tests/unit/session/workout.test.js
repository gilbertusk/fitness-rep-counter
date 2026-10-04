import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  createWorkout, startWorkout, stopWorkout, chooseExercise, setSpeech, forgetHistory, applyScores, stepFrame,
} from '../../../src/core/session/workout.js';
import { PHASE } from '../../../src/core/session/session.js';
import { poseStream, squatPose, random, gaussian } from '../../fixtures/syntheticPose.js';

/** Run a synthetic stream through the workout loop, collecting every effect. */
function play(state, stream, offsetMs = 0) {
  const spoken = [];
  const windows = [];
  const closed = [];
  let view = null;
  let s = state;
  stream.frames.forEach((frame, i) => {
    const out = stepFrame(s, { frame, timeMs: offsetMs + stream.timestampsMs[i], size: stream.size });
    s = out.state;
    view = out.view;
    if (out.effects.speak) spoken.push(out.effects.speak);
    windows.push(...out.effects.classify);
    closed.push(...out.effects.closedSets);
  });
  return { state: s, view, spoken, windows, closed };
}

const leaningSquat = (depth) => {
  const points = squatPose(depth);
  // Push the shoulders far forward: a torso about 60° from vertical at the bottom.
  const shift = 0.25 * depth;
  return { ...points, 0: [points[0][0] + shift, points[0][1]], 11: [points[11][0] + shift, points[11][1] + 0.08 * depth],
    12: [points[12][0] + shift, points[12][1] + 0.08 * depth] };
};

function staticStream(points, seconds, { fps = 30, jitter = 0.001, seed = 4 } = {}) {
  const noise = gaussian(random(seed));
  const frames = [];
  const timestampsMs = [];
  for (let i = 0; i < seconds * fps; i += 1) {
    frames.push(Array.from({ length: 33 }, (_, k) => (points[k]
      ? { x: points[k][0] + jitter * noise(), y: points[k][1] + jitter * noise(), z: 0, visibility: 0.9 }
      : { x: 0.5, y: 0.5, z: 0, visibility: 0.05 })));
    timestampsMs.push(Math.round((i * 1000) / fps));
  }
  return { frames, timestampsMs, size: { width: 1280, height: 720 } };
}

const PLANK = {
  0: [0.2, 0.47], 7: [0.22, 0.47], 8: [0.22, 0.47],
  11: [0.3, 0.5], 12: [0.31, 0.5], 13: [0.3, 0.62], 14: [0.31, 0.62], 15: [0.32, 0.72], 16: [0.33, 0.72],
  23: [0.55, 0.525], 24: [0.56, 0.525], 25: [0.68, 0.54], 26: [0.69, 0.54], 27: [0.8, 0.55], 28: [0.81, 0.55],
};

test('without a classifier, auto-detect is unavailable but reps are still counted', () => {
  const stream = poseStream({ segments: [{ reps: 4, repSeconds: 2 }] });
  const { view, state } = play(startWorkout(createWorkout(), 0), stream);
  assert.equal(view.autoAvailable, false);
  assert.equal(state.session.phase, PHASE.DETECTING);
  assert.equal(view.reps, 4);
  assert.equal(view.label, null);
});

test('a manual choice counts reps under that exercise and speaks each count', () => {
  const stream = poseStream({ segments: [{ reps: 4, repSeconds: 2.5 }] });
  const { view, spoken } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), stream);
  assert.equal(view.phase, PHASE.COUNTING);
  assert.equal(view.name, 'Squat');
  assert.equal(view.reps, 4);
  assert.deepEqual(spoken, ['1', '2', '3', '4']);
  assert.equal(view.status, 'experimental');
});

test('speech is rate limited and can be switched off', () => {
  const fast = poseStream({ segments: [{ reps: 6, repSeconds: 0.9 }] });
  const { spoken } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), fast);
  assert.ok(spoken.length < 6, `${spoken.length} utterances for six reps under one second each`);
  const silent = setSpeech(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), false);
  assert.deepEqual(play(silent, fast).spoken, []);
});

test('classifier windows come out about once a second, and two confident ones lock the label', () => {
  const labels = ['plank', 'push_up', 'squat'];
  let state = startWorkout(createWorkout({ classifierLabels: labels }), 0);
  const stream = poseStream({ segments: [{ reps: 4, repSeconds: 2 }] });
  const { windows, state: after } = play(state, stream);
  assert.ok(windows.length >= 7 && windows.length <= 10, `${windows.length} windows over ${stream.frames.length / 30} s`);
  assert.equal(windows[0].window.length, 30);
  state = after;
  const squatScores = [0, 0, 4];
  state = applyScores(applyScores(state, squatScores), squatScores);
  assert.equal(state.session.phase, PHASE.COUNTING);
  assert.equal(state.session.set.label, 'squat');
  const next = stepFrame(state, { frame: stream.frames.at(-1), timeMs: stream.timestampsMs.at(-1) + 33, size: stream.size });
  assert.ok(next.view.confidence > 0.9, `confidence ${next.view.confidence}`);
});

test('scores are ignored when there is no classifier', () => {
  const state = createWorkout();
  assert.equal(applyScores(state, [1, 2, 3]), state);
});

test('a form warning appears after half a second of bad form and is spoken once', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 3 }], pose: leaningSquat });
  const { view, spoken } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), stream);
  assert.equal(view.hasRules, true);
  assert.ok(spoken.includes('Punggung terlalu membungkuk'), spoken.join(' | '));
  assert.ok(view.reps >= 2);
});

test('the view says whether form could be judged at all, not just whether it broke a rule', () => {
  const stream = poseStream({ segments: [{ reps: 1, repSeconds: 2 }] });
  const seen = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), stream);
  assert.equal(seen.view.formChecked, true);
  const hidden = { ...stream, frames: stream.frames.map((f) => f.map((p) => ({ ...p, visibility: 0.1 }))) };
  const unseen = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), hidden);
  assert.equal(unseen.view.formChecked, false);
  assert.deepEqual(unseen.view.warnings, []);
});

test('a plank hold runs the timer and keeps the set open while still', () => {
  const { view, state } = play(chooseExercise(startWorkout(createWorkout(), 0), 'plank', 0), staticStream(PLANK, 8));
  assert.equal(view.task, 'hold');
  assert.equal(view.hold.holding, true);
  assert.ok(view.hold.bestMs > 7000, `${view.hold.bestMs} ms`);
  assert.equal(state.session.phase, PHASE.COUNTING, 'a still plank is not a rest');
});

test('stillness after a set closes it and reports it once', () => {
  const work = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], restSeconds: 1, tailSeconds: 5 });
  const { closed, state } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), work);
  assert.equal(closed.length, 1);
  assert.equal(closed[0].label, 'squat');
  assert.equal(closed[0].reps, 3);
  assert.equal(state.session.phase, PHASE.RESTING);
  assert.deepEqual(forgetHistory(state).session.sets, []);
});

test('a new set starts its counter from zero', () => {
  const work = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], restSeconds: 1, tailSeconds: 5 });
  const first = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), work);
  const second = play(first.state, poseStream({ segments: [{ reps: 2, repSeconds: 2 }], seed: 9 }), 20000);
  assert.equal(second.view.reps, 2);
  assert.equal(second.state.session.setId, first.state.session.setId + 1);
});

test('the view carries what the panel shows', () => {
  const stream = poseStream({ segments: [{ reps: 1, repSeconds: 2 }] });
  const { view } = play(startWorkout(createWorkout(), 0), stream);
  assert.ok(view.fps > 25 && view.fps < 35, `fps ${view.fps}`);
  assert.equal(view.landmarks.length, 33, 'smoothed landmarks for the overlay');
  assert.equal(view.quality, null);
  const lost = stepFrame(play(startWorkout(createWorkout(), 0), stream).state, { frame: null, timeMs: 99999, size: stream.size });
  assert.match(lost.view.quality, /tidak terdeteksi/);
});

test('stopping closes the running set', () => {
  const stream = poseStream({ segments: [{ reps: 2, repSeconds: 2 }], tailSeconds: 0 });
  const { state } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), stream);
  const stopped = stopWorkout(state, 99999);
  assert.equal(stopped.session.phase, PHASE.IDLE);
  assert.equal(stopped.session.sets.length, 1);
});

test('the view reports visible keypoints and how long the running set has lasted', () => {
  const stream = poseStream({ segments: [{ reps: 2, repSeconds: 2 }], tailSeconds: 0 });
  const { view } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), stream);
  assert.ok(view.visibleKeypoints > 0 && view.visibleKeypoints <= 33, `${view.visibleKeypoints}`);
  const last = stream.timestampsMs[stream.timestampsMs.length - 1];
  assert.ok(view.setDurationMs > 0 && view.setDurationMs <= last, `${view.setDurationMs} ms`);
  assert.equal(view.repsWithWarning, 0);
  assert.equal(view.rest, null, 'no rest while a set is running');
});

test('while resting, the view carries the finished set and a growing rest time', () => {
  const work = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], restSeconds: 1, tailSeconds: 5 });
  const { state, view } = play(chooseExercise(startWorkout(createWorkout(), 0), 'squat', 0), work);
  assert.equal(view.phase, PHASE.RESTING);
  assert.equal(view.rest.name, 'Squat');
  assert.equal(view.rest.reps, 3);
  assert.equal(view.setDurationMs, null);
  const later = stepFrame(state, { frame: null, timeMs: work.timestampsMs.at(-1) + 2000, size: work.size }).view;
  assert.ok(later.rest.sinceMs > view.rest.sinceMs && later.rest.sinceMs >= 2000, `${later.rest.sinceMs} ms`);
  assert.equal(later.visibleKeypoints, 0, 'no pose, no keypoints');
});
