import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  MATCH_TOLERANCE_MS, matchReps, videoMetrics, aggregate, aggregateByClass, summarize,
} from '../lib/repMetrics.js';

const close = (actual, expected, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) <= tolerance,
  `${actual} vs ${expected}`);

// ---------------------------------------------------------------- matching

test('reps within the tolerance pair up one to one', () => {
  const { pairs, unmatchedPredicted, unmatchedTruth } = matchReps([1100, 2950, 9000], [1000, 3000, 5000]);
  assert.deepEqual(pairs, [{ predicted: 1100, truth: 1000 }, { predicted: 2950, truth: 3000 }]);
  assert.deepEqual(unmatchedPredicted, [9000]);
  assert.deepEqual(unmatchedTruth, [5000]);
});

test('the tolerance is one second and inclusive', () => {
  assert.equal(MATCH_TOLERANCE_MS, 1000);
  assert.equal(matchReps([2000], [1000]).pairs.length, 1);
  assert.equal(matchReps([2001], [1000]).pairs.length, 0);
});

test('the closest pair wins when two predictions compete for one label', () => {
  // Hand-checked: 1900 is 100 ms from 2000, 1300 is 700 ms; only one may claim it.
  const { pairs, unmatchedPredicted } = matchReps([1300, 1900], [2000]);
  assert.deepEqual(pairs, [{ predicted: 1900, truth: 2000 }]);
  assert.deepEqual(unmatchedPredicted, [1300]);
});

test('a double count near one rep yields one match and one false positive', () => {
  const { pairs, unmatchedPredicted } = matchReps([3000, 3200], [3100]);
  assert.equal(pairs.length, 1);
  assert.equal(unmatchedPredicted.length, 1);
});

test('empty inputs match nothing', () => {
  assert.deepEqual(matchReps([], [1000]).unmatchedTruth, [1000]);
  assert.deepEqual(matchReps([1000], []).unmatchedPredicted, [1000]);
});

// ---------------------------------------------------------------- one video

test('a perfect count has zero error and full precision and recall', () => {
  const m = videoMetrics({ predicted: [1100, 2100, 3100], truth: [1000, 2000, 3000] });
  assert.equal(m.absError, 0);
  assert.equal(m.offByOne, true);
  assert.equal(m.normalizedError, 0);
  assert.equal(m.precision, 1);
  assert.equal(m.recall, 1);
  close(m.latencyMs, 100);
});

test('counting one rep too few is off by one, with recall 2/3', () => {
  const m = videoMetrics({ predicted: [1000, 2000], truth: [1000, 2000, 3000] });
  assert.equal(m.error, -1);
  assert.equal(m.absError, 1);
  assert.equal(m.offByOne, true);
  close(m.normalizedError, 1 / 3);
  assert.equal(m.precision, 1);
  close(m.recall, 2 / 3);
});

test('two reps too many is outside off-by-one', () => {
  const m = videoMetrics({ predicted: [1000, 1500, 2000, 2500], truth: [1000, 2000] });
  assert.equal(m.error, 2);
  assert.equal(m.offByOne, false);
  assert.equal(m.normalizedError, 1);
});

test('the right count at the wrong times is caught by precision and recall, not by MAE', () => {
  const m = videoMetrics({ predicted: [5000, 9000], truth: [1000, 2000] });
  assert.equal(m.absError, 0);
  assert.equal(m.precision, 0);
  assert.equal(m.recall, 0);
  assert.equal(m.latencyMs, null);
});

test('undefined ratios are null rather than zero', () => {
  const silent = videoMetrics({ predicted: [], truth: [1000] });
  assert.equal(silent.precision, null);
  assert.equal(silent.recall, 0);
  const noReps = videoMetrics({ predicted: [1000], truth: [] });
  assert.equal(noReps.normalizedError, null);
  assert.equal(noReps.recall, null);
});

// ---------------------------------------------------------------- many videos

const row = (label, predicted, truth, extra = {}) => ({ label, ...videoMetrics({ predicted, truth }), ...extra });

test('count metrics are averaged per video, rep metrics pooled over reps', () => {
  const rows = [
    row('squat', [1000, 2000], [1000, 2000]),                           // exact
    row('squat', [1000], [1000, 2000, 3000, 4000]),                     // 3 short
  ];
  const summary = aggregate(rows);
  assert.equal(summary.n, 2);
  close(summary.mae, 1.5);
  close(summary.obo, 0.5);
  close(summary.normalizedMae, (0 + 0.75) / 2);
  close(summary.precision, 3 / 3);
  close(summary.recall, 3 / 6);
});

test('an empty set aggregates to nulls instead of throwing', () => {
  assert.deepEqual(aggregate([]), {
    n: 0, mae: null, obo: null, normalizedMae: null, precision: null, recall: null, latencyMs: null,
  });
});

test('per-class aggregation keys by label in order', () => {
  const byClass = aggregateByClass([row('squat', [1], [1]), row('bench_press', [], [1000]), row('squat', [], [5])]);
  assert.deepEqual(Object.keys(byClass), ['bench_press', 'squat']);
  assert.equal(byClass.squat.n, 2);
  close(byClass.bench_press.mae, 1);
});

test('the summary reports with and without ambiguous videos', () => {
  const rows = [row('squat', [1000], [1000]), row('squat', [], [1000, 2000, 3000], { ambiguous: true })];
  const summary = summarize(rows);
  close(summary.all.mae, 1.5);
  close(summary.withoutAmbiguous.mae, 0);
  assert.equal(summary.withoutAmbiguous.n, 1);
});

test('videos a counter cannot handle are excluded and counted, never scored as zero', () => {
  const rows = [row('squat', [1000], [1000]), row('lat_pulldown', [], [1000, 2000], { status: 'n/a' })];
  const summary = summarize(rows);
  assert.equal(summary.all.n, 1);
  assert.equal(summary.notApplicable, 1);
  assert.equal('lat_pulldown' in summary.byClass, false);
});
