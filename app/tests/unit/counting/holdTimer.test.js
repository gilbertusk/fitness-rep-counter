import { test } from 'node:test';
import assert from 'node:assert/strict';

import { HOLD_DEFAULTS, createHoldTimer, updateHoldTimer, finishHold } from '../../../src/core/counting/holdTimer.js';

/** Feed (inPosition, timeMs) pairs at 30 fps-like steps. */
const feed = (steps, state = createHoldTimer()) => steps.reduce((s, [inPosition, t]) => updateHoldTimer(s, inPosition, t), state);
const span = (inPosition, fromMs, toMs, stepMs = 50) => {
  const steps = [];
  for (let t = fromMs; t <= toMs; t += stepMs) steps.push([inPosition, t]);
  return steps;
};

test('the timer runs while the body is in position', () => {
  const state = feed(span(true, 0, 5000));
  assert.equal(state.holding, true);
  assert.ok(Math.abs(state.currentMs - 5000) < 40);
  assert.equal(state.bestMs, state.currentMs);
});

test('a dropout shorter than the grace period does not end the hold', () => {
  assert.equal(HOLD_DEFAULTS.graceMs, 400);
  const state = feed([...span(true, 0, 3000), ...span(false, 3050, 3300), ...span(true, 3350, 6000)]);
  assert.equal(state.holding, true);
  assert.ok(state.currentMs > 5900, 'one continuous hold');
});

test('breaking position past the grace period ends and records the hold', () => {
  const state = feed([...span(true, 0, 3000), ...span(false, 3050, 4000)]);
  assert.equal(state.holding, false);
  assert.equal(state.currentMs, 0);
  assert.deepEqual(state.holds, [3000]);
  assert.equal(state.bestMs, 3000);
});

test('the best hold survives a later, shorter one', () => {
  const state = feed([...span(true, 0, 6000), ...span(false, 6050, 7000), ...span(true, 7050, 9000)]);
  assert.equal(state.bestMs, 6000);
  assert.ok(state.currentMs < state.bestMs);
});

test('a flicker shorter than a second is not recorded as a hold', () => {
  assert.deepEqual(feed([...span(true, 0, 500), ...span(false, 550, 1500)]).holds, []);
});

test('finishing closes a running hold and is harmless otherwise', () => {
  const running = feed(span(true, 0, 2000));
  assert.deepEqual(finishHold(running).holds, [running.lastGoodMs - running.startMs]);
  const idle = createHoldTimer();
  assert.equal(finishHold(idle), idle);
  assert.equal(updateHoldTimer(idle, false, 100), idle);
});

test('updating never mutates the previous state', () => {
  const before = feed(span(true, 0, 1000));
  const snapshot = JSON.stringify(before);
  updateHoldTimer(before, false, 5000);
  assert.equal(JSON.stringify(before), snapshot);
  assert.ok(Object.isFrozen(before));
});
