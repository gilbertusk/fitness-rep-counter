import { calculateAngle } from '../geometry/angles.js';

/**
 * Window features for the exercise classifier — implementation of `docs/FEATURES.md`.
 *
 * The same specification is implemented in `ml/src/repcount/features/features.py`; both are pinned
 * to `app/tests/fixtures/features_golden.json` by a parity test on each side (tolerance 1e-4).
 * Change one, change all three.
 *
 * Pure: no DOM, no browser API, no mutation of the caller's arrays.
 */

// MediaPipe indices of the 13 landmarks we keep, in the fixed order of docs/FEATURES.md §4.
export const LANDMARK_INDICES = Object.freeze([0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);

const L_SHOULDER = 1, R_SHOULDER = 2;
const L_ELBOW = 3, R_ELBOW = 4;
const L_WRIST = 5, R_WRIST = 6;
const L_HIP = 7, R_HIP = 8;
const L_KNEE = 9, R_KNEE = 10;
const L_ANKLE = 11, R_ANKLE = 12;

// Angle at b between b→a and b→c, in the fixed order of docs/FEATURES.md §5.
const ANGLE_TRIPLETS = Object.freeze([
  [L_SHOULDER, L_ELBOW, L_WRIST],
  [R_SHOULDER, R_ELBOW, R_WRIST],
  [L_ELBOW, L_SHOULDER, L_HIP],
  [R_ELBOW, R_SHOULDER, R_HIP],
  [L_SHOULDER, L_HIP, L_KNEE],
  [R_SHOULDER, R_HIP, R_KNEE],
  [L_HIP, L_KNEE, L_ANKLE],
  [R_HIP, R_KNEE, R_ANKLE],
]);

// Left↔right swaps for the flip augmentation (docs/FEATURES.md §9); nose maps to itself.
const FLIP_POINT_PAIRS = Object.freeze([
  [L_SHOULDER, R_SHOULDER], [L_ELBOW, R_ELBOW], [L_WRIST, R_WRIST],
  [L_HIP, R_HIP], [L_KNEE, R_KNEE], [L_ANKLE, R_ANKLE],
]);
const FLIP_ANGLE_PAIRS = Object.freeze([[0, 1], [2, 3], [4, 5], [6, 7]]);

export const N_POINTS = LANDMARK_INDICES.length;
export const N_ANGLES = ANGLE_TRIPLETS.length;
export const N_FEATURES = N_POINTS * 3 + N_ANGLES; // 13 × (x, y, visibility) + 8 angles = 47

export const TARGET_FPS = 15;
export const WINDOW_FRAMES = 30;
export const WINDOW_STRIDE = 15;
export const MAX_INTERPOLATION_GAP = 5;
export const MAX_MISSING_RATIO = 0.3;
const MIN_SCALE = 1e-6;

const x = (i) => i * 3;
const y = (i) => i * 3 + 1;
const vis = (i) => i * 3 + 2;

/** Mean of the values that are finite; NaN when neither is (mirrors numpy's nanmean of two). */
function meanOfPresent(a, b) {
  const hasA = Number.isFinite(a);
  const hasB = Number.isFinite(b);
  if (hasA && hasB) return (a + b) / 2;
  if (hasA) return a;
  return hasB ? b : NaN;
}

// ---------------------------------------------------------------- resampling

/**
 * Source frame index nearest to each 1/targetFps grid point (§2). Ties pick the lower index.
 * @param {ArrayLike<number>} timestampsMs ascending frame timestamps
 * @returns {number[]}
 */
export function resampleIndices(timestampsMs, targetFps = TARGET_FPS) {
  const t = timestampsMs;
  if (t.length === 0) return [];

  const step = 1000 / targetFps;
  const count = Math.floor((t[t.length - 1] - t[0]) / step) + 1;
  const indices = [];
  let after = 0; // first index whose timestamp is >= the current target; targets only move forward

  for (let k = 0; k < count; k += 1) {
    const target = t[0] + k * step;
    while (after < t.length && t[after] < target) after += 1;
    const lower = Math.min(Math.max(after - 1, 0), t.length - 1);
    const upper = Math.min(after, t.length - 1);
    indices.push(Math.abs(t[lower] - target) <= Math.abs(t[upper] - target) ? lower : upper);
  }
  return indices;
}

// ---------------------------------------------------------------- per-frame geometry

/**
 * One MediaPipe frame → 39 values of aspect-corrected x, y and visibility (§1, §4).
 * @param {Array<{x: number, y: number, visibility?: number}>|null|undefined} frame
 *   the 33 landmarks of the detected pose, or null/undefined when no pose was found
 * @returns {Float64Array} length 39, all NaN when the frame has no pose
 */
export function selectLandmarks(frame, width, height) {
  const points = new Float64Array(N_POINTS * 3).fill(NaN);
  if (!frame) return points;

  const aspect = width / height;
  for (let i = 0; i < N_POINTS; i += 1) {
    const landmark = frame[LANDMARK_INDICES[i]];
    if (!landmark) continue;
    points[x(i)] = landmark.x * aspect;
    points[y(i)] = landmark.y;
    points[vis(i)] = landmark.visibility ?? NaN;
  }
  return points;
}

/**
 * Center on the hip midpoint, scale by the mean shoulder–hip distance (§3).
 * A frame whose center or scale cannot be computed becomes all-NaN.
 */
export function normalizePoints(points) {
  const centerX = meanOfPresent(points[x(L_HIP)], points[x(R_HIP)]);
  const centerY = meanOfPresent(points[y(L_HIP)], points[y(R_HIP)]);
  const torsoLeft = Math.hypot(points[x(L_SHOULDER)] - points[x(L_HIP)], points[y(L_SHOULDER)] - points[y(L_HIP)]);
  const torsoRight = Math.hypot(points[x(R_SHOULDER)] - points[x(R_HIP)], points[y(R_SHOULDER)] - points[y(R_HIP)]);
  const scale = meanOfPresent(torsoLeft, torsoRight);

  const out = new Float64Array(N_POINTS * 3);
  if (!(Number.isFinite(scale) && scale >= MIN_SCALE && Number.isFinite(centerX) && Number.isFinite(centerY))) {
    return out.fill(NaN);
  }
  for (let i = 0; i < N_POINTS; i += 1) {
    out[x(i)] = (points[x(i)] - centerX) / scale;
    out[y(i)] = (points[y(i)] - centerY) / scale;
    out[vis(i)] = points[vis(i)];
  }
  return out;
}

/** Normalized points → 8 joint angles in [0, 1] (degrees / 180); NaN when undefined (§5). */
export function jointAngles(points) {
  const at = (i) => ({ x: points[x(i)], y: points[y(i)] });
  const angles = new Float64Array(N_ANGLES);
  ANGLE_TRIPLETS.forEach(([a, b, c], i) => {
    // calculateAngle reports an undefined angle as null; the feature spec uses NaN for that.
    const degrees = calculateAngle(at(a), at(b), at(c));
    angles[i] = degrees === null ? NaN : degrees / 180;
  });
  return angles;
}

/** Normalized points → one 47-value feature row in the fixed layout of §6. */
export function frameFeatures(points) {
  const row = new Float64Array(N_FEATURES);
  row.set(points, 0);
  row.set(jointAngles(points), N_POINTS * 3);
  return row;
}

// ---------------------------------------------------------------- gaps and windows

/** Linearly fill NaN runs of at most maxGap frames, per column, never extrapolating (§7). */
export function interpolateGaps(sequence, maxGap = MAX_INTERPOLATION_GAP) {
  const out = sequence.map((row) => Float64Array.from(row));
  for (let column = 0; column < N_FEATURES; column += 1) {
    let previous = -1;
    for (let i = 0; i < out.length; i += 1) {
      if (!Number.isFinite(out[i][column])) continue;
      const gap = i - previous - 1;
      if (previous >= 0 && gap > 0 && gap <= maxGap) {
        const slope = (out[i][column] - out[previous][column]) / (i - previous);
        for (let k = previous + 1; k < i; k += 1) out[k][column] = slope * (k - previous) + out[previous][column];
      }
      previous = i;
    }
  }
  return out;
}

/** A frame is missing when any of its features is still NaN (§7). */
export function missingFrames(sequence) {
  return sequence.map((row) => !row.every(Number.isFinite));
}

export function windowStarts(nFrames, length = WINDOW_FRAMES, stride = WINDOW_STRIDE) {
  const starts = [];
  for (let start = 0; start + length <= nFrames; start += stride) starts.push(start);
  return starts;
}

/**
 * Cut a feature sequence into windows (§8).
 * @returns {{ windows: Float64Array[][], starts: number[], missingRatio: number[] }}
 *   windows have their remaining NaN replaced by 0; missingRatio drives dropping or flagging
 */
export function makeWindows(sequence, length = WINDOW_FRAMES, stride = WINDOW_STRIDE) {
  const starts = windowStarts(sequence.length, length, stride);
  const missing = missingFrames(sequence);
  return {
    starts,
    windows: starts.map((start) => sequence.slice(start, start + length)
      .map((row) => row.map((value) => (Number.isFinite(value) ? value : 0)))),
    missingRatio: starts.map((start) => missing.slice(start, start + length).filter(Boolean).length / length),
  };
}

/** Windows kept for training and evaluation; the rest are only flagged at inference (§8). */
export function usableWindows(missingRatio, maxRatio = MAX_MISSING_RATIO) {
  return missingRatio.map((ratio) => ratio <= maxRatio);
}

// ---------------------------------------------------------------- pipeline and augmentation

/**
 * Raw MediaPipe frames → feature rows at TARGET_FPS with gaps filled (§1–§7).
 * @param {Array<Array<object>|null|undefined>} landmarks one entry per source frame
 * @param {ArrayLike<number>} timestampsMs matching frame timestamps
 * @returns {Float64Array[]}
 */
export function sequenceFeatures(landmarks, timestampsMs, width, height) {
  const indices = resampleIndices(timestampsMs);
  const rows = indices.map((i) => frameFeatures(normalizePoints(selectLandmarks(landmarks[i], width, height))));
  return interpolateGaps(rows);
}

/** Mirror left↔right: swap paired landmarks and angles, negate normalized x (§9). */
export function flipFeatures(window) {
  return window.map((row) => {
    const out = Float64Array.from(row);
    FLIP_POINT_PAIRS.forEach(([left, right]) => {
      for (let offset = 0; offset < 3; offset += 1) {
        out[left * 3 + offset] = row[right * 3 + offset];
        out[right * 3 + offset] = row[left * 3 + offset];
      }
    });
    FLIP_ANGLE_PAIRS.forEach(([left, right]) => {
      out[N_POINTS * 3 + left] = row[N_POINTS * 3 + right];
      out[N_POINTS * 3 + right] = row[N_POINTS * 3 + left];
    });
    for (let i = 0; i < N_POINTS; i += 1) out[x(i)] *= -1;
    return out;
  });
}
