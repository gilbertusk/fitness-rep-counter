import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  ROW_COLUMNS, GRIDS, labelsForSplit, expandGrid, scoreVideo, toCsvRow, pickTraceVideos, formatSummary, bench, main,
} from '../evalReps.js';
import { parseCsv, toCsv, REP_LABEL_COLUMNS } from '../../labeler/src/labels.js';
import { summarize } from '../lib/repMetrics.js';
import { poseStream } from '../../../app/tests/fixtures/syntheticPose.js';

const label = (video_id, labelName, times, extra = {}) => ({
  video_id, label: labelName, rep_count: String(times.length), rep_timestamps_ms: times.join(';'),
  hold_start_ms: '', hold_end_ms: '', is_ambiguous: 'false', notes: '', labeler: 'gk', labeled_at: '2026-10-02T09:00:00Z',
  ...extra,
});

// ---------------------------------------------------------------- pure helpers

test('a split scores its repetition videos only: never plank, never another split', () => {
  const rows = [label('squat_1', 'squat', [1000]), label('squat_2', 'squat', [1000]),
    label('plank_1', 'plank', [], { rep_count: '', hold_start_ms: '0', hold_end_ms: '9000' })];
  const doc = { splits: { val: ['squat_1', 'plank_1'], test: ['squat_2'] } };
  assert.deepEqual(labelsForSplit(rows, 'val', doc).map((r) => r.video_id), ['squat_1']);
  assert.deepEqual(labelsForSplit(rows, 'train', doc), []);
});

test('a grid expands to every combination', () => {
  assert.deepEqual(expandGrid({ a: [1, 2], b: [3] }), [{ a: 1, b: 3 }, { a: 2, b: 3 }]);
  assert.equal(expandGrid(GRIDS.generic).length, 18);
  assert.deepEqual(expandGrid({}), [{}]);
});

test('a scored video carries its metrics; an unsupported one is marked n/a', () => {
  const row = label('squat_1', 'squat', [1000, 2000]);
  const ok = scoreVideo(row, { count: 2, repTimes: [1100, 2050], msPerFrame: 0.1 });
  assert.equal(ok.status, 'ok');
  assert.equal(ok.absError, 0);
  const na = scoreVideo(row, null);
  assert.equal(na.status, 'n/a');
  assert.equal(summarize([ok, na]).notApplicable, 1);
});

test('CSV rows hold every column, and n/a rows leave the prediction blank', () => {
  const row = label('squat_1', 'squat', [1000, 2000], { is_ambiguous: 'true' });
  const ok = toCsvRow(scoreVideo(row, { count: 1, repTimes: [1100], msPerFrame: 0.12345 }));
  assert.deepEqual(Object.keys(ok), [...ROW_COLUMNS]);
  assert.equal(ok.predicted_count, 1);
  assert.equal(ok.ambiguous, 'true');
  assert.equal(ok.ms_per_frame, '0.1235');
  assert.equal(ok.truth_ms, '1000;2000');
  const na = toCsvRow(scoreVideo(row, null));
  assert.equal(na.predicted_count, '');
  assert.equal(na.precision, '');
});

test('the figures show one clear success and the worst misses', () => {
  const s = (id, absError, truthCount) => ({ video_id: id, status: 'ok', absError, truthCount });
  const picked = pickTraceVideos([s('a', 0, 3), s('b', 0, 8), s('c', 4, 5), s('d', 1, 5), { status: 'n/a' }]);
  assert.deepEqual(picked.map((p) => p.video_id), ['b', 'c', 'd']);
  assert.deepEqual(pickTraceVideos([s('x', 2, 1)]).map((p) => p.video_id), ['x'], 'no success → only the misses');
});

test('the terminal summary shows both subsets, every class and the N/A count', () => {
  const rows = [scoreVideo(label('squat_1', 'squat', [1000]), { repTimes: [1000], msPerFrame: 0 }),
    scoreVideo(label('lat_pulldown_1', 'lat_pulldown', [1000]), null)];
  const text = formatSummary('threshold', 'val', summarize(rows));
  assert.match(text, /without ambiguous/);
  assert.match(text, /squat/);
  assert.match(text, /1 videos N\/A/);
});

test('the bench reports per-frame cost for every counter', () => {
  const results = bench(['generic', 'naive-peaks'], 4);
  assert.deepEqual(results.map((r) => r.name), ['generic', 'naive-peaks']);
  results.forEach((r) => {
    assert.ok(r.frames > 0);
    assert.ok(r.meanMs >= 0 && r.p95Ms >= 0 && r.maxMs >= r.p95Ms);
  });
});

// ---------------------------------------------------------------- the CLI end to end

let dir;

before(() => {
  dir = mkdtempSync(join(tmpdir(), 'evalreps-'));
  mkdirSync(join(dir, 'keypoints_json'));
  const rows = [];
  [['squat_1', 3], ['squat_2', 2]].forEach(([id, reps], i) => {
    const stream = poseStream({ segments: [{ reps, repSeconds: 2 }], seed: 30 + i, jitter: 0.0005 });
    writeFileSync(join(dir, 'keypoints_json', `${id}.json`), JSON.stringify({
      video_id: id, label: 'squat', width: 1280, height: 720, timestamps_ms: stream.timestampsMs,
      landmarks: stream.frames.map((frame) => frame.map((p) => [p.x, p.y, p.z, p.visibility])),
    }));
    rows.push(label(id, 'squat', stream.repTimesMs));
  });
  rows.push(label('squat_9', 'squat', [1000]));  // labelled but never extracted
  writeFileSync(join(dir, 'rep_labels.csv'), toCsv(rows, REP_LABEL_COLUMNS));
  writeFileSync(join(dir, 'split.json'), JSON.stringify({ splits: { val: ['squat_1', 'squat_2', 'squat_9'], test: [] } }));
  process.env.FITNESS_DATA_DIR = join(dir, 'data');
});

after(() => {
  delete process.env.FITNESS_DATA_DIR;
  rmSync(dir, { recursive: true, force: true });
});

const cli = (...extra) => main(['--labels', join(dir, 'rep_labels.csv'), '--split-file', join(dir, 'split.json'),
  '--keypoints-dir', join(dir, 'keypoints_json'), '--out-dir', join(dir, 'reports'), ...extra]);

test('a full run writes the per-video CSV and the traces for the figures', () => {
  assert.equal(cli('--counter', 'generic'), 0);
  const rows = parseCsv(readFileSync(join(dir, 'reports', 'reps_generic_val.csv'), 'utf8'));
  assert.deepEqual(rows.map((r) => r.video_id), ['squat_1', 'squat_2'], 'squat_9 has no keypoints');
  assert.deepEqual(rows.map((r) => r.abs_error), ['0', '0']);
  const traces = JSON.parse(readFileSync(join(dir, 'data', 'runs', 'rep_traces_generic_val.json'), 'utf8'));
  assert.ok(traces.traces.length >= 1);
  assert.equal(traces.traces[0].signal.length, traces.traces[0].timestamps_ms.length);
});

test('the threshold counter runs through the same harness', () => {
  assert.equal(cli('--counter', 'threshold', '--no-traces'), 0);
  assert.ok(existsSync(join(dir, 'reports', 'reps_threshold_val.csv')));
});

test('a grid run tunes on val and writes its ranking', () => {
  assert.equal(cli('--counter', 'naive-peaks', '--grid'), 0);
  const ranking = parseCsv(readFileSync(join(dir, 'reports', 'grid_naive-peaks_val.csv'), 'utf8'));
  assert.equal(ranking.length, expandGrid(GRIDS['naive-peaks']).length);
  assert.ok(Number(ranking[0].mae) <= Number(ranking.at(-1).mae), 'best first');
});

test('the guards refuse what would make the numbers dishonest', () => {
  assert.throws(() => cli('--split', 'test'), /scored once, at the end: add --final/);
  assert.throws(() => cli('--split', 'test', '--final', '--grid'), /--grid tunes on val only/);
  assert.throws(() => cli('--counter', 'threshold', '--grid'), /--grid tunes on val only/);
  assert.throws(() => cli('--counter', 'magic'), /unknown counter magic/);
  assert.throws(() => main(['--labels', join(dir, 'nope.csv')]), /stage 2 labels come first/);
});

test('the final test run is allowed once tuning is declared finished', () => {
  assert.equal(cli('--split', 'test', '--final', '--no-traces'), 0);
});
