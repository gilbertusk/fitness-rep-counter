import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createCounter, updateCounter, PHASE, FEEDBACK } from '../src/core/repCounter.js';

const CONFIG = { downAngle: 90, upAngle: 160, partialAngle: 130 };
const run = (angles) => angles.reduce((state, angle) => updateCounter(state, angle, CONFIG), createCounter());

test('starts at zero in the UP phase', () => {
  const state = createCounter();
  assert.equal(state.count, 0);
  assert.equal(state.phase, PHASE.UP);
});

test('counts one full repetition', () => {
  const state = run([170, 120, 85, 120, 165]);
  assert.equal(state.count, 1);
  assert.equal(state.feedback, FEEDBACK.GOOD);
});

test('counts multiple repetitions', () => {
  assert.equal(run([170, 80, 170, 80, 170, 80, 170]).count, 3);
});

test('does not count while still in the DOWN phase', () => {
  const state = run([170, 80, 120]);
  assert.equal(state.count, 0);
  assert.equal(state.phase, PHASE.DOWN);
});

test('jitter around the down threshold does not double count', () => {
  assert.equal(run([170, 89, 91, 88, 92, 170]).count, 1);
});

test('flags a partial repetition without counting it', () => {
  const state = run([170, 120, 170]);
  assert.equal(state.count, 0);
  assert.equal(state.feedback, FEEDBACK.PARTIAL);
});

test('a small dip is not flagged as partial', () => {
  const state = run([170, 150, 170]);
  assert.equal(state.feedback, null);
});

test('ignores missing angles', () => {
  assert.equal(run([170, null, 80, Number.NaN, 170]).count, 1);
});

test('never mutates the previous state', () => {
  const before = createCounter();
  updateCounter(before, 80, CONFIG);
  assert.equal(before.phase, PHASE.UP);
  assert.ok(Object.isFrozen(before));
});
