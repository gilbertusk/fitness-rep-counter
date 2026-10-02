import { test } from 'node:test';
import assert from 'node:assert/strict';

import { DEBOUNCE_MS, createDebounce, updateDebounce } from '../../../src/core/form/debounce.js';

const feed = (frames) => frames.reduce(({ state }, [ids, t]) => updateDebounce(state, ids, t), { state: createDebounce() });

test('a warning appears only after half a second without a break', () => {
  assert.equal(DEBOUNCE_MS, 500);
  assert.deepEqual(feed([[['lean'], 0], [['lean'], 499]]).visible, []);
  assert.deepEqual(feed([[['lean'], 0], [['lean'], 250], [['lean'], 500]]).visible, ['lean']);
});

test('a single bad frame never shows a warning', () => {
  assert.deepEqual(feed([[['lean'], 0], [[], 33], [['lean'], 600]]).visible, [], 'the clock restarted at 600');
});

test('a warning disappears as soon as the rule holds again', () => {
  assert.deepEqual(feed([[['lean'], 0], [['lean'], 600], [[], 633]]).visible, []);
});

test('each warning keeps its own clock and the frame order', () => {
  const { visible } = feed([[['a'], 0], [['a', 'b'], 300], [['b', 'a'], 600]]);
  assert.deepEqual(visible, ['a']);
  assert.deepEqual(feed([[['a'], 0], [['a', 'b'], 300], [['b', 'a'], 800]]).visible, ['b', 'a']);
});

test('updating never mutates the previous state', () => {
  const first = updateDebounce(createDebounce(), ['lean'], 0);
  const snapshot = JSON.stringify(first.state);
  updateDebounce(first.state, ['other'], 100);
  assert.equal(JSON.stringify(first.state), snapshot);
  assert.ok(Object.isFrozen(first.state));
});
