import { test } from 'node:test';
import assert from 'node:assert/strict';

import { ONE_EURO_DEFAULTS, filterValue, filterPose } from '../../../src/core/geometry/oneEuroFilter.js';
import { random, gaussian } from '../../fixtures/syntheticPose.js';

const run = (values, dtMs = 33.3, options) => {
  let state = null;
  return values.map((x, i) => {
    const out = filterValue(state, x, i * dtMs, options);
    state = out.state;
    return out.value;
  });
};

const spread = (values) => Math.max(...values) - Math.min(...values);

test('the first sample passes through unchanged', () => {
  assert.equal(filterValue(null, 0.42, 0).value, 0.42);
});

test('jitter on a still point is cut by more than half', () => {
  const noise = gaussian(random(5));
  const raw = Array.from({ length: 300 }, () => 0.5 + 0.004 * noise());
  const smoothed = run(raw).slice(30);
  assert.ok(spread(smoothed) < 0.5 * spread(raw.slice(30)), `${spread(smoothed)} vs ${spread(raw.slice(30))}`);
});

test('a fast movement is followed with little lag', () => {
  // A 1 Hz swing of ±0.2: the speed term opens the filter, so the output tracks closely.
  const raw = Array.from({ length: 120 }, (_, i) => 0.5 + 0.2 * Math.sin((2 * Math.PI * i) / 30));
  const smoothed = run(raw);
  const worst = Math.max(...raw.slice(30).map((x, i) => Math.abs(x - smoothed[i + 30])));
  assert.ok(worst < 0.06, `largest gap ${worst}`);
});

test('a higher beta follows motion more closely, trading off smoothness', () => {
  const raw = Array.from({ length: 120 }, (_, i) => 0.5 + 0.2 * Math.sin((2 * Math.PI * i) / 30));
  const lag = (beta) => {
    const out = run(raw, 33.3, { ...ONE_EURO_DEFAULTS, beta });
    return Math.max(...raw.slice(30).map((x, i) => Math.abs(x - out[i + 30])));
  };
  assert.ok(lag(50) < lag(0));
});

test('missing values and repeated timestamps never break the filter', () => {
  const first = filterValue(null, 0.5, 100);
  assert.ok(Number.isNaN(filterValue(first.state, NaN, 133).value));
  assert.equal(filterValue(first.state, NaN, 133).state, first.state);
  assert.equal(filterValue(first.state, 0.7, 100).value, 0.7, 'no time step → restart, no division by zero');
});

test('a pose filters x, y and z and leaves visibility alone', () => {
  const pose = (x) => Array.from({ length: 33 }, () => ({ x, y: x, z: x, visibility: 0.77 }));
  const first = filterPose(null, pose(0.5), 0);
  const second = filterPose(first.state, pose(0.6), 33);
  assert.equal(second.landmarks.length, 33);
  assert.ok(second.landmarks[0].x > 0.5 && second.landmarks[0].x < 0.6, 'pulled towards the new value');
  assert.equal(second.landmarks[0].visibility, 0.77);
});

test('a frame without a pose keeps the filter state for when the pose comes back', () => {
  const first = filterPose(null, [{ x: 0.5, y: 0.5, z: 0 }], 0);
  const gap = filterPose(first.state, null, 33);
  assert.equal(gap.landmarks, null);
  assert.equal(gap.state, first.state);
});

test('filtering never mutates the input pose', () => {
  const input = [{ x: 0.5, y: 0.5, z: 0, visibility: 1 }];
  const snapshot = JSON.stringify(input);
  const first = filterPose(null, input, 0);
  filterPose(first.state, [{ x: 0.9, y: 0.9, z: 0, visibility: 1 }], 33);
  assert.equal(JSON.stringify(input), snapshot);
});
