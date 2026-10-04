import { jointAngles, normalizePoints, selectLandmarks } from '../features/features.js';

/**
 * The floor for stage 3: count peaks of whichever joint angle has varied most since the start, with
 * one fixed prominence and one fixed minimum gap for every exercise. No resampling, no adaptive
 * thresholds, no idea of a resting posture. A generic counter that cannot beat this is not earning
 * its complexity. Streaming, pure and immutable, with the same call shape as genericCounter.js.
 */

export const NAIVE_DEFAULTS = Object.freeze({
  prominenceDegrees: 15, // a peak must stand this far above the lowest point since the last one
  minGapSeconds: 0.5,
});

const N_ANGLES = 8;

export function createNaivePeakCounter(options = {}) {
  return Object.freeze({
    config: { ...NAIVE_DEFAULTS, ...options },
    n: 0,
    means: Object.freeze(new Array(N_ANGLES).fill(0)),
    m2: Object.freeze(new Array(N_ANGLES).fill(0)),  // Welford sums of squares, per angle
    channel: null,
    previous: Object.freeze([]),                    // the last two values of the chosen angle, with times
    lowest: Infinity,
    lastPeakMs: -Infinity,
    count: 0,
    repTimes: Object.freeze([]),
  });
}

/** Running variance per angle (Welford), so the busiest angle can be picked without a buffer. */
function accumulate(state, angles) {
  const n = state.n + 1;
  const means = state.means.map((mean, a) => mean + (angles[a] - mean) / n);
  const m2 = state.m2.map((sum, a) => sum + (angles[a] - state.means[a]) * (angles[a] - means[a]));
  const channel = m2.indexOf(Math.max(...m2));
  return { n, means: Object.freeze(means), m2: Object.freeze(m2), channel };
}

/**
 * @param {Array<object>|null|undefined} frame MediaPipe's 33 landmarks, or nothing when no pose
 * @param {number} timestampMs frame time
 * @param {{ width: number, height: number }} size source frame size
 */
export function updateNaivePeakCounter(state, frame, timestampMs, size) {
  const angles = Array.from(jointAngles(normalizePoints(selectLandmarks(frame, size.width, size.height))), (a) => a * 180);
  if (!angles.every(Number.isFinite)) return state;

  const stats = accumulate(state, angles);
  // The naive part: when the busiest joint changes, the history simply restarts on the new one.
  const history = stats.channel === state.channel ? state.previous : [];
  const value = angles[stats.channel];
  const lowest = stats.channel === state.channel ? Math.min(state.lowest, value) : value;
  const previous = Object.freeze([...history, { value, timeMs: timestampMs }].slice(-3));
  let next = { ...state, ...stats, previous, lowest };

  if (previous.length === 3) {
    const [before, peak, after] = previous;
    const isPeak = peak.value > before.value && peak.value >= after.value;
    const prominent = peak.value - state.lowest >= state.config.prominenceDegrees;
    const spaced = peak.timeMs - state.lastPeakMs >= 1000 * state.config.minGapSeconds;
    if (isPeak && prominent && spaced) {
      next = { ...next, count: state.count + 1, repTimes: Object.freeze([...state.repTimes, peak.timeMs]),
        lastPeakMs: peak.timeMs, lowest: after.value };
    }
  }
  return Object.freeze(next);
}
