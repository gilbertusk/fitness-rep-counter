import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { CATALOG, STATUS, exerciseName } from '../../src/core/exercises.js';

const SPLIT = JSON.parse(readFileSync(new URL('../../../ml/splits/split_v1.json', import.meta.url), 'utf8'));

test('the catalog lists exactly the 22 dataset classes', () => {
  assert.deepEqual(Object.keys(CATALOG).sort(), Object.keys(SPLIT.per_class).sort());
});

test('only plank is a hold; everything else counts repetitions', () => {
  assert.deepEqual(Object.entries(CATALOG).filter(([, e]) => e.task === 'hold').map(([label]) => label), ['plank']);
});

test('nothing is marked supported before stage 3 has evaluated it', () => {
  assert.ok(Object.values(CATALOG).every((e) => e.status === STATUS.EXPERIMENTAL));
});

test('names fall back to the label for anything unknown', () => {
  assert.equal(exerciseName('push_up'), 'Push-up');
  assert.equal(exerciseName('handstand'), 'handstand');
});
