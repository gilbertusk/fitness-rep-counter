import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_THRESHOLDS, UNCERTAIN,
  softmax, createClassifier, updateClassifier, resetClassifier,
} from '../../../src/core/classify/classifier.js';

const LABELS = ['squat', 'push_up', 'plank'];
const classifier = (thresholds = {}) => createClassifier({ labels: LABELS, thresholds });

/** Scores that softmax turns into (roughly) the requested probabilities. */
const scoresFor = (probabilities) => probabilities.map(Math.log);

test('softmax returns a probability distribution', () => {
  const probabilities = softmax([2, 1, 0]);
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(probabilities.every((p) => p > 0 && p < 1));
  assert.ok(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
});

test('softmax survives large scores without overflowing', () => {
  const probabilities = softmax([1000, 999, 998]);
  assert.ok(probabilities.every(Number.isFinite));
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
});

test('softmax is unchanged by a constant offset', () => {
  softmax([2, 1, 0]).forEach((value, i) => assert.ok(Math.abs(value - softmax([12, 11, 10])[i]) < 1e-12));
});

test('creating a classifier needs labels', () => {
  assert.throws(() => createClassifier({ labels: [] }), /non-empty labels/);
  assert.throws(() => createClassifier({}), /non-empty labels/);
});

test('defaults are filled in and overrides respected', () => {
  assert.deepEqual(classifier().thresholds, DEFAULT_THRESHOLDS);
  assert.equal(classifier({ minConfidence: 0.9 }).thresholds.minConfidence, 0.9);
  assert.equal(classifier({ minConfidence: 0.9 }).thresholds.smoothing, DEFAULT_THRESHOLDS.smoothing);
});

test('a confident window yields a confident prediction', () => {
  const state = updateClassifier(classifier(), scoresFor([0.9, 0.05, 0.05]));
  assert.equal(state.prediction.label, 'squat');
  assert.ok(Math.abs(state.prediction.confidence - 0.9) < 1e-9);
  assert.equal(state.prediction.uncertain, false);
  assert.equal(state.nWindows, 1);
});

test('a low top probability is reported as uncertain', () => {
  const state = updateClassifier(classifier(), scoresFor([0.4, 0.35, 0.25]));
  assert.equal(state.prediction.uncertain, true);
  assert.equal(state.prediction.uncertainReason, UNCERTAIN.LOW_CONFIDENCE);
});

test('two close classes are reported as a close call', () => {
  const state = updateClassifier(classifier({ minConfidence: 0.3 }), scoresFor([0.52, 0.45, 0.03]));
  assert.equal(state.prediction.uncertain, true);
  assert.equal(state.prediction.uncertainReason, UNCERTAIN.CLOSE_CALL);
  assert.ok(Math.abs(state.prediction.margin - 0.07) < 1e-9);
});

test('a window with too many missing frames is uncertain however confident the model is', () => {
  const state = updateClassifier(classifier(), scoresFor([0.99, 0.005, 0.005]), { poorPose: true });
  assert.equal(state.prediction.uncertain, true);
  assert.equal(state.prediction.uncertainReason, UNCERTAIN.POOR_POSE);
  assert.equal(state.prediction.label, 'squat', 'the label is still reported, only flagged');
});

test('poor pose outranks the other uncertainty reasons', () => {
  const state = updateClassifier(classifier(), scoresFor([0.4, 0.35, 0.25]), { poorPose: true });
  assert.equal(state.prediction.uncertainReason, UNCERTAIN.POOR_POSE);
});

test('smoothing pulls the average towards the newer window', () => {
  const first = updateClassifier(classifier({ smoothing: 0.5 }), scoresFor([0.9, 0.05, 0.05]));
  const second = updateClassifier(first, scoresFor([0.1, 0.8, 0.1]));
  assert.ok(Math.abs(second.probabilities[0] - 0.5) < 1e-9);
  assert.ok(Math.abs(second.probabilities[1] - 0.425) < 1e-9);
  assert.equal(second.nWindows, 2);
});

test('smoothing lets a sustained class overtake a single stale window', () => {
  let state = updateClassifier(classifier({ smoothing: 0.4 }), scoresFor([0.9, 0.05, 0.05]));
  assert.equal(state.prediction.label, 'squat');
  for (let i = 0; i < 3; i += 1) state = updateClassifier(state, scoresFor([0.05, 0.9, 0.05]));
  assert.equal(state.prediction.label, 'push_up');
});

test('one odd window does not flip a settled prediction', () => {
  let state = classifier({ smoothing: 0.3 });
  for (let i = 0; i < 5; i += 1) state = updateClassifier(state, scoresFor([0.9, 0.05, 0.05]));
  const afterGlitch = updateClassifier(state, scoresFor([0.05, 0.9, 0.05]));
  assert.equal(afterGlitch.prediction.label, 'squat');
});

test('updating never mutates the previous state', () => {
  const first = updateClassifier(classifier(), scoresFor([0.9, 0.05, 0.05]));
  const snapshot = [...first.probabilities];
  updateClassifier(first, scoresFor([0.05, 0.05, 0.9]));
  assert.deepEqual(first.probabilities, snapshot);
  assert.equal(first.nWindows, 1);
});

test('a mismatched score count is rejected rather than silently mislabelled', () => {
  assert.throws(() => updateClassifier(classifier(), [0.5, 0.5]), /2 scores for 3 labels/);
});

test('resetting forgets the smoothed history', () => {
  const state = updateClassifier(classifier(), scoresFor([0.9, 0.05, 0.05]));
  const fresh = resetClassifier(state);
  assert.equal(fresh.probabilities, null);
  assert.equal(fresh.nWindows, 0);
  assert.equal(fresh.prediction, undefined);
  assert.deepEqual(fresh.labels, LABELS);
});

test('ties are broken by label order so the prediction is deterministic', () => {
  const state = updateClassifier(classifier({ minConfidence: 0.1, minMargin: 0 }), scoresFor([0.5, 0.5, 0.0]));
  assert.equal(state.prediction.label, 'squat');
});
