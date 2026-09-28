import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  LANDMARK_INDICES, N_FEATURES, N_POINTS, MAX_MISSING_RATIO,
  TARGET_FPS, WINDOW_FRAMES, WINDOW_STRIDE, MAX_INTERPOLATION_GAP,
  resampleIndices, selectLandmarks, normalizePoints, jointAngles, frameFeatures,
  interpolateGaps, missingFrames, windowStarts, makeWindows, usableWindows,
  sequenceFeatures, flipFeatures,
} from '../../../src/core/features/features.js';

const GOLDEN = JSON.parse(readFileSync(new URL('../../fixtures/features_golden.json', import.meta.url), 'utf8'));

/** JSON has no NaN literal, so the fixture stores it as null. */
const num = (value) => (value === null ? NaN : value);

/** One fixture frame of 33 [x, y, z, visibility] rows → the landmark objects MediaPipe hands us. */
const toFrame = (rows) => rows.map(([lx, ly, lz, lv]) => ({ x: num(lx), y: num(ly), z: num(lz), visibility: num(lv) }));

function assertClose(actual, expected, tolerance, path = '') {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, `length differs at ${path}`);
    expected.forEach((value, i) => assertClose(actual[i], value, tolerance, `${path}[${i}]`));
    return;
  }
  const want = num(expected);
  if (Number.isNaN(want)) {
    assert.ok(Number.isNaN(actual), `expected NaN at ${path}, got ${actual}`);
    return;
  }
  assert.ok(Math.abs(actual - want) <= tolerance, `at ${path}: ${actual} vs ${want} (tol ${tolerance})`);
}

const STANDING = {
  0: [0.5, 0.1],
  11: [0.45, 0.25], 12: [0.55, 0.25],
  13: [0.45, 0.4], 14: [0.55, 0.4],
  15: [0.45, 0.55], 16: [0.55, 0.55],
  23: [0.46, 0.55], 24: [0.54, 0.55],
  25: [0.46, 0.75], 26: [0.54, 0.75],
  27: [0.46, 0.95], 28: [0.54, 0.95],
};

function poseFrame(points, visibility = 1) {
  return Array.from({ length: 33 }, (_, i) => (
    points[i] ? { x: points[i][0], y: points[i][1], z: 0, visibility } : { x: NaN, y: NaN, z: NaN, visibility: NaN }
  ));
}

const anglesOf = (points, width = 640, height = 640) =>
  Array.from(jointAngles(normalizePoints(selectLandmarks(poseFrame(points), width, height))), (a) => a * 180);

// ---------------------------------------------------------------- parity with the golden fixture

test('constants match the fixture the Python implementation produced', () => {
  assert.deepEqual(GOLDEN.constants, {
    targetFps: TARGET_FPS,
    windowFrames: WINDOW_FRAMES,
    windowStride: WINDOW_STRIDE,
    maxInterpolationGap: MAX_INTERPOLATION_GAP,
    maxMissingRatio: MAX_MISSING_RATIO,
    nFeatures: N_FEATURES,
  });
});

test('resampling reproduces the golden indices', () => {
  assert.deepEqual(resampleIndices(GOLDEN.input.timestampsMs), GOLDEN.expected.resampleIndices);
});

test('the pipeline reproduces the golden feature sequence', () => {
  const { landmarks, timestampsMs, width, height } = GOLDEN.input;
  const features = sequenceFeatures(landmarks.map(toFrame), timestampsMs, width, height);

  assert.equal(features.length, GOLDEN.expected.sequenceFeatures.length);
  features.forEach((row, i) => assertClose(Array.from(row), GOLDEN.expected.sequenceFeatures[i],
    GOLDEN.tolerance, `frame ${i}`));
});

test('the pipeline reproduces the golden missing-frame flags', () => {
  const { landmarks, timestampsMs, width, height } = GOLDEN.input;
  const features = sequenceFeatures(landmarks.map(toFrame), timestampsMs, width, height);
  assert.deepEqual(missingFrames(features), GOLDEN.expected.missing);
});

test('the pipeline reproduces the golden windows', () => {
  const { landmarks, timestampsMs, width, height } = GOLDEN.input;
  const cut = makeWindows(sequenceFeatures(landmarks.map(toFrame), timestampsMs, width, height));

  assert.deepEqual(cut.starts, GOLDEN.expected.windowStarts);
  assert.deepEqual(usableWindows(cut.missingRatio), GOLDEN.expected.usable);
  assertClose(cut.missingRatio, GOLDEN.expected.missingRatio, GOLDEN.tolerance, 'missingRatio');
  cut.windows.forEach((window, w) => window.forEach((row, i) =>
    assertClose(Array.from(row), GOLDEN.expected.windows[w][i], GOLDEN.tolerance, `window ${w} frame ${i}`)));
});

test('flipping reproduces the golden mirrored window', () => {
  const { landmarks, timestampsMs, width, height } = GOLDEN.input;
  const cut = makeWindows(sequenceFeatures(landmarks.map(toFrame), timestampsMs, width, height));
  flipFeatures(cut.windows[0]).forEach((row, i) =>
    assertClose(Array.from(row), GOLDEN.expected.flippedFirstWindow[i], GOLDEN.tolerance, `flipped frame ${i}`));
});

// ---------------------------------------------------------------- resampling

test('a 30 fps sequence is halved', () => {
  const timestamps = Array.from({ length: 60 }, (_, i) => Math.round((i * 1000) / 30));
  assert.deepEqual(resampleIndices(timestamps), Array.from({ length: 30 }, (_, i) => i * 2));
});

test('a 15 fps sequence keeps every frame', () => {
  const timestamps = Array.from({ length: 20 }, (_, i) => Math.round((i * 1000) / 15));
  assert.deepEqual(resampleIndices(timestamps), Array.from({ length: 20 }, (_, i) => i));
});

test('an empty or single-frame sequence resamples without throwing', () => {
  assert.deepEqual(resampleIndices([]), []);
  assert.deepEqual(resampleIndices([17]), [0]);
});

test('a stalled source reuses the nearest frame', () => {
  assert.deepEqual(resampleIndices([0, 200, 400]), [0, 0, 1, 1, 1, 2, 2]);
});

// ---------------------------------------------------------------- geometry

test('only x is stretched by the aspect ratio', () => {
  const points = selectLandmarks(poseFrame(STANDING), 1280, 720);
  assert.equal(points.length, N_POINTS * 3);
  assertClose(points[0], 0.5 * (1280 / 720), 1e-12, 'nose x');
  assertClose(points[1], 0.1, 1e-12, 'nose y');
});

test('a frame without a pose becomes all NaN', () => {
  assert.ok(Array.from(selectLandmarks(null, 640, 480)).every(Number.isNaN));
  assert.ok(Array.from(selectLandmarks(undefined, 640, 480)).every(Number.isNaN));
});

test('a landmark without visibility reads as NaN rather than zero', () => {
  const frame = poseFrame(STANDING);
  delete frame[LANDMARK_INDICES[0]].visibility;
  assert.ok(Number.isNaN(selectLandmarks(frame, 640, 640)[2]));
});

test('normalizing puts the hip midpoint at the origin', () => {
  const points = normalizePoints(selectLandmarks(poseFrame(STANDING), 640, 640));
  assertClose((points[7 * 3] + points[8 * 3]) / 2, 0, 1e-12, 'hip mid x');
  assertClose((points[7 * 3 + 1] + points[8 * 3 + 1]) / 2, 0, 1e-12, 'hip mid y');
});

test('normalizing is invariant to translation and zoom', () => {
  const moved = Object.fromEntries(Object.entries(STANDING).map(([i, [px, py]]) => [i, [0.3 + 0.5 * px, 0.1 + 0.5 * py]]));
  const original = normalizePoints(selectLandmarks(poseFrame(STANDING), 640, 640));
  const zoomed = normalizePoints(selectLandmarks(poseFrame(moved), 640, 640));
  original.forEach((value, i) => assertClose(zoomed[i], value, 1e-9, `value ${i}`));
});

test('a collapsed torso marks the frame missing', () => {
  const collapsed = { ...STANDING, 11: STANDING[23], 12: STANDING[24] };
  assert.ok(Array.from(normalizePoints(selectLandmarks(poseFrame(collapsed), 640, 640))).every(Number.isNaN));
});

test('one missing side still yields a usable frame', () => {
  const oneSide = { ...STANDING };
  delete oneSide[12];
  delete oneSide[24];
  assert.ok(Number.isFinite(normalizePoints(selectLandmarks(poseFrame(oneSide), 640, 640))[7 * 3]));
});

test('straight limbs measure 180 degrees', () => {
  const angles = anglesOf(STANDING);
  assertClose(angles[0], 180, 0.5, 'left elbow');
  assertClose(angles[6], 180, 0.5, 'left knee');
});

test('a right angle at the elbow measures 90 degrees', () => {
  assertClose(anglesOf({ ...STANDING, 15: [0.6, 0.4], 16: [0.7, 0.4] })[0], 90, 0.5, 'left elbow');
});

test('an angle is NaN when a point is missing or two points coincide', () => {
  const withoutWrist = { ...STANDING };
  delete withoutWrist[15];
  assert.ok(Number.isNaN(anglesOf(withoutWrist)[0]));
  assert.ok(Number.isNaN(anglesOf({ ...STANDING, 15: STANDING[13] })[0]));
});

test('aspect correction changes a diagonal limb', () => {
  const splayed = { ...STANDING, 27: [0.3, 0.95], 28: [0.7, 0.95] };
  assert.ok(Math.abs(anglesOf(splayed, 640, 640)[6] - anglesOf(splayed, 1280, 720)[6]) > 0.5);
});

test('a feature row follows the layout of the spec', () => {
  const points = normalizePoints(selectLandmarks(poseFrame(STANDING), 640, 640));
  const row = frameFeatures(points);
  assert.equal(row.length, N_FEATURES);
  assert.equal(N_FEATURES, 47);
  points.forEach((value, i) => assertClose(row[i], value, 1e-12, `point ${i}`));
  jointAngles(points).forEach((value, i) => assertClose(row[N_POINTS * 3 + i], value, 1e-12, `angle ${i}`));
});

// ---------------------------------------------------------------- gaps, windows, flip

const column = (values) => values.map((v) => { const row = new Float64Array(N_FEATURES); row[0] = v; return row; });

test('gaps up to five frames are filled linearly', () => {
  const values = [0, 1, 2, NaN, NaN, NaN, NaN, NaN, 8, 9];
  const filled = interpolateGaps(column(values));
  [3, 4, 5, 6, 7].forEach((i) => assertClose(filled[i][0], i, 1e-12, `frame ${i}`));
});

test('gaps longer than five frames stay missing', () => {
  const values = [0, 1, 2, NaN, NaN, NaN, NaN, NaN, NaN, 9];
  const filled = interpolateGaps(column(values));
  [3, 4, 5, 6, 7, 8].forEach((i) => assert.ok(Number.isNaN(filled[i][0]), `frame ${i} should stay NaN`));
});

test('leading and trailing gaps are never extrapolated', () => {
  const filled = interpolateGaps(column([NaN, NaN, 2, 3, 4, NaN, NaN]));
  [0, 1, 5, 6].forEach((i) => assert.ok(Number.isNaN(filled[i][0]), `frame ${i} should stay NaN`));
});

test('interpolation does not mutate the caller sequence', () => {
  const source = column([0, NaN, 2]);
  interpolateGaps(source);
  assert.ok(Number.isNaN(source[1][0]));
});

test('missing frames flag any remaining NaN', () => {
  const rows = column([0, 0, 0]);
  rows[1][7] = NaN;
  assert.deepEqual(missingFrames(rows), [false, true, false]);
});

test('window starts follow the stride and need a full window', () => {
  assert.deepEqual(windowStarts(48), [0, 15]);
  assert.deepEqual(windowStarts(29), []);
  assert.deepEqual(windowStarts(30), [0]);
});

test('windows replace leftover NaN with zero and report the missing ratio', () => {
  const rows = Array.from({ length: 30 }, () => new Float64Array(N_FEATURES));
  rows.slice(0, 3).forEach((row) => row.fill(NaN));
  const cut = makeWindows(rows);
  assert.equal(cut.windows.length, 1);
  assert.ok(cut.windows[0].every((row) => row.every(Number.isFinite)));
  assertClose(cut.missingRatio[0], 0.1, 1e-12, 'missing ratio');
});

test('a window is usable exactly at the thirty percent boundary', () => {
  assert.deepEqual(usableWindows([0.3, 0.3001]), [true, false]);
});

test('flipping swaps sides, negates x and is its own inverse', () => {
  const row = frameFeatures(normalizePoints(selectLandmarks(poseFrame(STANDING), 640, 640)));
  const [flipped] = flipFeatures([row]);
  assertClose(flipped[1 * 3], -row[2 * 3], 1e-12, 'left shoulder x');
  assertClose(flipped[39], row[40], 1e-12, 'left elbow angle');
  const [back] = flipFeatures([flipped]);
  row.forEach((value, i) => assertClose(back[i], value, 1e-12, `value ${i}`));
});

test('flipping a mirrored pose reproduces the original', () => {
  const swap = { 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27 };
  const relabelled = Object.fromEntries(
    Object.entries(STANDING).map(([i, [px, py]]) => [swap[i] ?? i, [1 - px, py]]),
  );
  const original = frameFeatures(normalizePoints(selectLandmarks(poseFrame(STANDING), 640, 640)));
  const other = frameFeatures(normalizePoints(selectLandmarks(poseFrame(relabelled), 640, 640)));
  const [restored] = flipFeatures([other]);
  for (let i = 0; i < N_POINTS * 3; i += 1) assertClose(restored[i], original[i], 1e-9, `value ${i}`);
});

test('an empty video produces no features', () => {
  assert.deepEqual(sequenceFeatures([], [], 640, 480), []);
});
