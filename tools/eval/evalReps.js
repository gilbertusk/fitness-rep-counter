#!/usr/bin/env node
/**
 * Stream labelled videos through a rep counter, frame by frame, and score it against the human labels.
 *
 *   node tools/eval/evalReps.js --counter generic --split val
 *   node tools/eval/evalReps.js --counter generic --split val --grid        # tuning, val only
 *   node tools/eval/evalReps.js --counter generic --split test --final      # once, at the end
 *   node tools/eval/evalReps.js --bench                                     # ms per frame, no data needed
 *
 * Reads data/keypoints_json/<video_id>.json (python -m repcount.export.keypoints_json),
 * labels/rep_labels.csv and ml/splits/split_v1.json. Writes reports/03-rep-counter/.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

import { COUNTERS, runCounter, toLandmarks } from './lib/counters.js';
import { summarize, videoMetrics } from './lib/repMetrics.js';
import { parseCsv, toCsv } from '../labeler/src/labels.js';
import { poseStream } from '../../app/tests/fixtures/syntheticPose.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
// Read on every call, not at import, so FITNESS_DATA_DIR set later (tests, scripts) is honoured.
const dataDir = () => (process.env.FITNESS_DATA_DIR ? resolve(process.env.FITNESS_DATA_DIR) : join(ROOT, 'data'));

export const ROW_COLUMNS = Object.freeze([
  'video_id', 'label', 'ambiguous', 'status', 'truth_count', 'predicted_count', 'error', 'abs_error',
  'off_by_one', 'precision', 'recall', 'latency_ms', 'ms_per_frame', 'truth_ms', 'predicted_ms',
]);

// Small grids, so tuning stays a few dozen runs over val and cannot quietly overfit it.
export const GRIDS = Object.freeze({
  generic: { smoothing: [0.35, 0.5, 0.65], minAmplitude: [0.1, 0.2, 0.4], lowZone: [0.25, 0.3] },
  'generic-angle': { smoothing: [0.35, 0.5, 0.65], minAmplitude: [0.04, 0.08, 0.16], lowZone: [0.25, 0.3] },
  'naive-peaks': { prominenceDegrees: [10, 15, 20, 25], minGapSeconds: [0.4, 0.5, 0.7] },
});

// ---------------------------------------------------------------- pure helpers

const parseTimes = (text) => String(text ?? '').split(';').filter((t) => t.trim() !== '').map(Number);

/** The label rows that this split scores: repetition videos only (plank is a hold, see labels/README.md). */
export function labelsForSplit(labelRows, split, splitDocument) {
  const ids = new Set(splitDocument.splits?.[split] ?? []);
  return labelRows.filter((row) => ids.has(row.video_id) && row.label !== 'plank' && row.rep_count !== '');
}

/** Every combination of a grid, e.g. { a: [1, 2], b: [3] } → [{a: 1, b: 3}, {a: 2, b: 3}]. */
export function expandGrid(grid) {
  return Object.entries(grid).reduce(
    (combos, [key, values]) => combos.flatMap((combo) => values.map((value) => ({ ...combo, [key]: value }))), [{}]);
}

/** One CSV row and one metrics row for a video, or an N/A row when the counter does not apply. */
export function scoreVideo(labelRow, result) {
  const truth = parseTimes(labelRow.rep_timestamps_ms);
  const base = { video_id: labelRow.video_id, label: labelRow.label, ambiguous: labelRow.is_ambiguous === 'true' };
  if (!result) {
    return { ...base, status: 'n/a', truthCount: truth.length, truth, predicted: [] };
  }
  return { ...base, status: 'ok', ...videoMetrics({ predicted: result.repTimes, truth }),
    truth, predicted: result.repTimes, msPerFrame: result.msPerFrame };
}

const fixed = (value, digits = 3) => (value === null || value === undefined ? '' : Number(value).toFixed(digits));

export function toCsvRow(scored) {
  return {
    video_id: scored.video_id, label: scored.label, ambiguous: String(scored.ambiguous), status: scored.status,
    truth_count: scored.truthCount ?? scored.truth.length, predicted_count: scored.status === 'ok' ? scored.predictedCount : '',
    error: scored.error ?? '', abs_error: scored.absError ?? '', off_by_one: scored.offByOne === undefined ? '' : String(scored.offByOne),
    precision: fixed(scored.precision), recall: fixed(scored.recall), latency_ms: fixed(scored.latencyMs, 0),
    ms_per_frame: fixed(scored.msPerFrame, 4), truth_ms: scored.truth.join(';'), predicted_ms: scored.predicted.join(';'),
  };
}

/** One clear success and the two worst misses, for the figures in the report. */
export function pickTraceVideos(scored, n = 3) {
  const ok = scored.filter((s) => s.status === 'ok');
  const success = ok.filter((s) => s.absError === 0).sort((a, b) => b.truthCount - a.truthCount)[0];
  const worst = ok.filter((s) => s !== success).sort((a, b) => b.absError - a.absError || b.truthCount - a.truthCount);
  return [success, ...worst].filter(Boolean).slice(0, n);
}

export function formatSummary(name, split, summary) {
  const line = (label, s) => `${label.padEnd(24)} ${String(s.n).padStart(4)}  ${fixed(s.mae, 2).padStart(6)}  `
    + `${fixed(s.obo, 2).padStart(5)}  ${fixed(s.precision, 2).padStart(5)}  ${fixed(s.recall, 2).padStart(5)}`;
  return [
    `${name} on ${split}`,
    `${'subset'.padEnd(24)} ${'n'.padStart(4)}  ${'MAE'.padStart(6)}  ${'OBO'.padStart(5)}  ${'prec'.padStart(5)}  ${'rec'.padStart(5)}`,
    line('all', summary.all),
    line('without ambiguous', summary.withoutAmbiguous),
    ...Object.entries(summary.byClass).map(([label, s]) => line(`  ${label}`, s)),
    ...(summary.notApplicable ? [`(${summary.notApplicable} videos N/A: this counter has no setting for their class)`] : []),
  ].join('\n');
}

// ---------------------------------------------------------------- I/O

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function loadInputs(args) {
  const labelsPath = resolve(args.labels);
  if (!existsSync(labelsPath)) {
    throw new Error(`${labelsPath} not found — stage 2 labels come first (labels/README.md)`);
  }
  const rows = labelsForSplit(parseCsv(readFileSync(labelsPath, 'utf8')), args.split, readJson(resolve(args['split-file'])));
  const videos = [];
  const missing = [];
  for (const row of rows) {
    const path = join(resolve(args['keypoints-dir']), `${row.video_id}.json`);
    if (existsSync(path)) videos.push({ row, video: readJson(path) });
    else missing.push(row.video_id);
  }
  return { videos, missing };
}

function evaluate(name, videos, params) {
  return videos.map(({ row, video }) => scoreVideo(row, runCounter(name, video, params)));
}

function traceVideo(name, video, params) {
  const counter = COUNTERS[name];
  const size = { width: video.width, height: video.height };
  let state = counter.create(video.label, params);
  const signal = video.landmarks.map((frame, i) => {
    state = counter.update(state, toLandmarks(frame), video.timestamps_ms[i], size);
    return state.signal ?? state.previous?.at(-1)?.value ?? null;
  });
  return { timestamps_ms: video.timestamps_ms, signal };
}

function writeTraces(name, split, videos, scored, params) {
  const byId = new Map(videos.map(({ video }) => [video.video_id, video]));
  const traces = pickTraceVideos(scored).map((s) => ({
    video_id: s.video_id, label: s.label, abs_error: s.absError, truth_ms: s.truth, predicted_ms: s.predicted,
    ...traceVideo(name, byId.get(s.video_id), params),
  }));
  const path = join(dataDir(), 'runs', `rep_traces_${name}_${split}.json`);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify({ counter: name, split, traces }) + '\n');
  return path;
}

function runGrid(name, videos, outDir) {
  const results = expandGrid(GRIDS[name]).map((params) => {
    const { all } = summarize(evaluate(name, videos, params));
    return { params: JSON.stringify(params), n: all.n, mae: fixed(all.mae), obo: fixed(all.obo),
      precision: fixed(all.precision), recall: fixed(all.recall) };
  }).sort((a, b) => Number(a.mae) - Number(b.mae) || Number(b.obo) - Number(a.obo));
  const path = join(outDir, `grid_${name}_val.csv`);
  writeFileSync(path, toCsv(results, ['params', 'n', 'mae', 'obo', 'precision', 'recall']));
  console.log(`${results.length} settings on val → ${path}\nbest five (MAE, OBO):`);
  results.slice(0, 5).forEach((r) => console.log(`  ${r.mae}  ${r.obo}  --params '${r.params}'`));
}

/** Per-frame cost on a synthetic one-minute stream: the "< 1 ms per frame" check, no dataset needed. */
export function bench(names = Object.keys(COUNTERS), seconds = 60) {
  const stream = poseStream({ segments: [{ reps: Math.floor(seconds / 2), repSeconds: 2 }], restSeconds: 0, tailSeconds: 0 });
  return names.map((name) => {
    const counter = COUNTERS[name];
    let state = counter.create('squat', {});
    const timings = stream.frames.map((frame, i) => {
      const started = performance.now();
      state = counter.update(state, frame, stream.timestampsMs[i], stream.size);
      return performance.now() - started;
    });
    const sorted = [...timings].sort((a, b) => a - b);
    return { name, frames: timings.length, meanMs: timings.reduce((a, b) => a + b, 0) / timings.length,
      p95Ms: sorted[Math.floor(0.95 * (sorted.length - 1))], maxMs: sorted.at(-1) };
  });
}

const options = () => ({
  counter: { type: 'string', default: 'generic' },
  split: { type: 'string', default: 'val' },
  final: { type: 'boolean', default: false },
  grid: { type: 'boolean', default: false },
  bench: { type: 'boolean', default: false },
  params: { type: 'string', default: '{}' },
  labels: { type: 'string', default: join(ROOT, 'labels', 'rep_labels.csv') },
  'split-file': { type: 'string', default: join(ROOT, 'ml', 'splits', 'split_v1.json') },
  'keypoints-dir': { type: 'string', default: join(dataDir(), 'keypoints_json') },
  'out-dir': { type: 'string', default: join(ROOT, 'reports', '03-rep-counter') },
  traces: { type: 'boolean', default: true },  // --no-traces to skip the figure data
});

export function main(argv = process.argv.slice(2)) {
  const { values: args } = parseArgs({ args: argv, options: options(), allowNegative: true });
  if (args.bench) {
    bench().forEach((b) => console.log(`${b.name.padEnd(14)} mean ${b.meanMs.toFixed(4)} ms  p95 ${b.p95Ms.toFixed(4)} ms  `
      + `max ${b.maxMs.toFixed(3)} ms  over ${b.frames} frames`));
    return 0;
  }
  if (!COUNTERS[args.counter]) throw new Error(`unknown counter ${args.counter}; known: ${Object.keys(COUNTERS).join(', ')}`);
  if (args.split === 'test' && !args.final) {
    throw new Error('the test split is scored once, at the end: add --final when tuning on val is finished');
  }
  if (args.grid && (args.split !== 'val' || !GRIDS[args.counter])) {
    throw new Error(`--grid tunes on val only, for: ${Object.keys(GRIDS).join(', ')}`);
  }

  const params = JSON.parse(args.params);
  const { videos, missing } = loadInputs(args);
  missing.forEach((id) => console.warn(`missing keypoints, not scored: ${id}`));
  mkdirSync(resolve(args['out-dir']), { recursive: true });
  if (args.grid) { runGrid(args.counter, videos, resolve(args['out-dir'])); return 0; }

  const scored = evaluate(args.counter, videos, params);
  const out = join(resolve(args['out-dir']), `reps_${args.counter}_${args.split}.csv`);
  writeFileSync(out, toCsv(scored.map(toCsvRow), ROW_COLUMNS));
  console.log(formatSummary(args.counter, args.split, summarize(scored)));
  console.log(`→ ${out}${missing.length ? ` (${missing.length} missing)` : ''}`);
  if (args.traces && scored.some((s) => s.status === 'ok')) {
    console.log(`traces → ${writeTraces(args.counter, args.split, videos, scored, params)}`);
  }
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    process.exitCode = main();
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
