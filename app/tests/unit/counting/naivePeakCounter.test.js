import { test } from 'node:test';
import assert from 'node:assert/strict';

import { NAIVE_DEFAULTS, createNaivePeakCounter, updateNaivePeakCounter } from '../../../src/core/counting/naivePeakCounter.js';
import { poseStream, curlPose } from '../../fixtures/syntheticPose.js';

const run = (stream, options) => stream.frames.reduce(
  (state, frame, i) => updateNaivePeakCounter(state, frame, stream.timestampsMs[i], stream.size), createNaivePeakCounter(options));

test('clean synthetic squats and curls are counted', () => {
  assert.equal(run(poseStream({ segments: [{ reps: 4, repSeconds: 2 }], jitter: 0.0005 })).count, 4);
  assert.equal(run(poseStream({ segments: [{ reps: 5, repSeconds: 1.5 }], pose: curlPose, jitter: 0.0005 })).count, 5);
});

test('standing still with near-perfect tracking counts nothing', () => {
  assert.equal(run(poseStream({ segments: [], restSeconds: 6, jitter: 0.0005 })).count, 0);
});

test('jitter on a straight limb fakes a peak: why this counter is only a floor', () => {
  // An elbow near 180° can only read lower under noise, so its spread is one-sided and wide
  // (163.6°–178.9° here at jitter 0.002) and clears the fixed 15° prominence. The generic counter's
  // adaptive range passes the same test at twice this jitter (genericCounter.test.js).
  assert.ok(run(poseStream({ segments: [], restSeconds: 6, jitter: 0.002 })).count > 0);
});

test('a peak needs the fixed prominence: a shallow wobble is not a rep', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], jitter: 0.0005 });
  assert.equal(run(stream, { prominenceDegrees: 90 }).count, 0);
  assert.equal(NAIVE_DEFAULTS.prominenceDegrees, 15);
});

test('frames without a pose leave the state as it was', () => {
  const state = createNaivePeakCounter();
  assert.equal(updateNaivePeakCounter(state, null, 0, { width: 640, height: 480 }), state);
});

test('updating never mutates the previous state', () => {
  const stream = poseStream({ segments: [{ reps: 2, repSeconds: 1 }], jitter: 0.0005 });
  const half = run({ ...stream, frames: stream.frames.slice(0, 40), timestampsMs: stream.timestampsMs.slice(0, 40) });
  const snapshot = JSON.stringify(half);
  updateNaivePeakCounter(half, stream.frames[40], stream.timestampsMs[40], stream.size);
  assert.equal(JSON.stringify(half), snapshot);
  assert.ok(Object.isFrozen(half));
});
