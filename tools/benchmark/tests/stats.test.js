import { test } from 'node:test';
import assert from 'node:assert/strict';

import { percentile, summarize, framesPerSecond, groupAssets, toMarkdown } from '../src/stats.js';

test('percentile interpolates between closest ranks and ignores non-finite samples', () => {
  assert.equal(percentile([1, 2, 3, 4, 5], 50), 3);
  assert.equal(percentile([4, 1, 3, 2], 50), 2.5);
  assert.equal(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 95), 10.5);
  assert.equal(percentile([7, NaN, Infinity], 95), 7);
  assert.ok(Number.isNaN(percentile([], 50)));
});

test('summarize gives n, median, p95, mean and max', () => {
  const s = summarize([10, 20, 30, 40, NaN]);
  assert.deepEqual({ n: s.n, median: s.median, mean: s.mean, max: s.max }, { n: 4, median: 25, mean: 25, max: 40 });
  assert.ok(Math.abs(s.p95 - 38.5) < 1e-9);
  assert.equal(summarize([]).n, 0);
});

test('frames per second counts intervals, not frames', () => {
  assert.equal(framesPerSecond([0, 100, 200, 300]), 10);
  assert.ok(Number.isNaN(framesPerSecond([5])));
  assert.ok(Number.isNaN(framesPerSecond([5, 5])));
});

test('assets are grouped by kind; unreadable sizes are flagged, not counted as zero', () => {
  const groups = groupAssets([
    { name: 'https://storage.googleapis.com/m/pose_landmarker_lite.task', encodedBodySize: 0, decodedBodySize: 0 },
    { name: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm/vision_wasm_internal.wasm', encodedBodySize: 3_000_000, decodedBodySize: 9_000_000 },
    { name: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs', encodedBodySize: 100, decodedBodySize: 400 },
    { name: 'http://localhost:5174/app/src/main.js?v=1', encodedBodySize: 50, decodedBodySize: 50 },
    { name: 'http://localhost:5174/tools/benchmark/src/bench.js', encodedBodySize: 50, decodedBodySize: 50 },
  ]);
  assert.deepEqual(groups.map((g) => [g.label, g.files, g.encodedBytes, g.unknown]), [
    ['Model pose (.task)', 1, 0, 1],
    ['MediaPipe WASM', 1, 3_000_000, 0],
    ['MediaPipe JS', 1, 100, 0],
    ['Kode app (JS/CSS/HTML/JSON)', 1, 50, 0],
  ]);
});

test('the Markdown report carries every measured number', () => {
  const markdown = toMarkdown({
    dateMs: Date.UTC(2026, 9, 2),
    device: { label: 'Laptop uji', userAgent: 'UA', cores: 8, delegate: 'GPU', renderer: 'ANGLE' },
    video: { name: 'push-up_17.mp4', width: 1920, height: 1080, durationS: 3.04, repeat: 2, fps: 30 },
    load: { poseReadyMs: 1234.5, firstPoseMs: 1500, classifierReadyMs: null },
    live: { fps: 29.4, frames: 180, posesFound: 175 },
    stages: [{ label: 'Pose', summary: { n: 180, median: 12.25, p95: 20, max: 31 } },
      { label: 'Counter', summary: { n: 180, median: 0.08, p95: 0.11, max: 0.5 } }],
    assets: [{ label: 'Model pose (.task)', files: 1, encodedBytes: 5_777_746, decodedBytes: 5_777_746, unknown: 0 }],
    notes: ['catatan'],
  });
  for (const expected of ['Laptop uji — 2026-10-02', '**29.4**', '1234.5 ms', 'tidak dimuat', '| Pose | 180 | 12.3 | 20.0 | 31.0 |',
    '| Counter | 180 | 0.080 | 0.110 | 0.500 |', '| Model pose (.task) | 1 | 5.78 | 5.78 |', '> catatan']) {
    assert.ok(markdown.includes(expected), `missing ${expected}\n${markdown}`);
  }
});
