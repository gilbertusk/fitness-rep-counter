import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SPEECH_GAP_MS, createSpeechQueue, say, nextUtterance, setSpeechEnabled } from '../../../src/core/session/speechQueue.js';

test('the first message is spoken right away', () => {
  const { text } = nextUtterance(say(createSpeechQueue(), 'satu'), 0);
  assert.equal(text, 'satu');
});

test('at most one utterance every two seconds', () => {
  assert.equal(SPEECH_GAP_MS, 2000);
  let state = say(createSpeechQueue(), 'satu');
  ({ state } = nextUtterance(state, 0));
  state = say(state, 'dua');
  assert.equal(nextUtterance(state, 1999).text, null);
  assert.equal(nextUtterance(state, 2000).text, 'dua');
});

test('only the newest pending message is kept', () => {
  let state = say(createSpeechQueue(), 'satu');
  ({ state } = nextUtterance(state, 0));
  state = say(say(say(state, 'dua'), 'tiga'), 'empat');
  assert.equal(nextUtterance(state, 2000).text, 'empat');
});

test('nothing is spoken twice', () => {
  let state = say(createSpeechQueue(), 'satu');
  ({ state } = nextUtterance(state, 0));
  assert.equal(nextUtterance(state, 5000).text, null);
});

test('speech can be switched off, and nothing stale is spoken when it comes back', () => {
  let state = say(createSpeechQueue(), 'satu');
  state = setSpeechEnabled(state, false);
  assert.equal(nextUtterance(state, 0).text, null);
  assert.equal(say(state, 'dua').pending, null);
  state = setSpeechEnabled(state, true);
  assert.equal(nextUtterance(state, 0).text, null);
  assert.equal(nextUtterance(say(state, 'tiga'), 0).text, 'tiga');
});

test('an empty message is ignored', () => {
  const state = createSpeechQueue();
  assert.equal(say(state, ''), state);
});

test('updates never mutate the previous state', () => {
  const state = say(createSpeechQueue(), 'satu');
  nextUtterance(state, 0);
  assert.equal(state.pending, 'satu');
  assert.ok(Object.isFrozen(state));
});
