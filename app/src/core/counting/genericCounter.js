import { frameFeatures, normalizePoints, selectLandmarks, N_POINTS, TARGET_FPS } from '../features/features.js';

/**
 * One rep counter for every exercise, with no per-exercise thresholds. Streaming: each call sees one
 * frame and never looks ahead. Pure and immutable: every update returns a new frozen state.
 *
 *   frame → motion vector (docs/FEATURES.md, x/y + angles) → resample to 15 fps → EMA
 *         → running window → one signal over the whole window → zone state machine → reps
 *
 * The whole window is re-projected on every sample with the current signal definition (PCA axis
 * and mean, or the chosen joint), so its range, its period and its "home" side always describe one
 * consistent signal rather than a mix of past definitions.
 *
 * A rep is counted when the signal comes back to the side it started on — the moment
 * labels/README.md asks humans to mark. Counting never waits for a period estimate: the clips in this
 * dataset average 7.8 s and often hold one to three reps, so a counter that needs two cycles to lock
 * on would miss the first rep of nearly every video. The autocorrelation period only sets the
 * minimum gap for shallow swings; a swing that reaches the far end of the range always counts, so a
 * set that speeds up is not throttled by the slower period still sitting in the window.
 *
 * Every value in DEFAULTS is a starting point, not a tuned result; tuning happens on the validation
 * split with `node tools/eval/evalReps.js --grid` (docs/PLAN.md §8).
 */

export const SIGNAL = Object.freeze({ PCA: 'pca', ANGLE: 'angle' });

export const DEFAULTS = Object.freeze({
  signal: SIGNAL.PCA,
  sampleFps: TARGET_FPS,       // the feature spec's rate
  bufferSeconds: 4,            // running window; periods up to half of it can be detected
  // EMA weight of the newest sample (≈ 130 ms time constant at 15 fps). Its gain is 0.81 on a 0.8 s
  // rep and 0.96 on a 2 s one; 0.35 would have been 0.64 vs 0.90, making fast reps look shallow.
  smoothing: 0.5,
  homeSeconds: 0.5,            // "home" is read from the mean of the window's first half second
  homeRegrowth: 2,             // … and read again if the range more than doubles before any rep
  lowZone: 0.3,                // below lo + 0.3·range is the low zone …
  highZone: 0.7,               // … above lo + 0.7·range the high zone; between them nothing changes
  fullSwing: 0.85,             // a swing this far across the range is never held back by the gap
  minAmplitude: { [SIGNAL.PCA]: 0.2, [SIGNAL.ANGLE]: 0.08 }, // range below this = standing still
  minPeriodSeconds: 0.5,
  minPeriodConfidence: 0.4,    // normalized autocorrelation needed to trust a period
  gapFraction: 0.6,            // shallow swings need ≥ 0.6 × period since the last rep
  fallbackGapSeconds: 0.4,     // the gap before a period is known, and the floor for every rep
  channelSwitchRatio: 1.5,     // angle signal: switch joints only when another varies 1.5× more
  axisResetDot: 0.8,           // PCA axis turning further than this means a new movement
  powerIterations: 6,
});

const ZONE = Object.freeze({ LOW: 'low', HIGH: 'high' });
// x and y of the 13 landmarks, then the 8 angles: visibility is not motion.
const MOTION_COLUMNS = Object.freeze([
  ...Array.from({ length: N_POINTS }, (_, i) => [i * 3, i * 3 + 1]).flat(),
  ...Array.from({ length: 8 }, (_, i) => N_POINTS * 3 + i),
]);
const ANGLE_OFFSET = N_POINTS * 2; // angles start here inside a motion vector

// ---------------------------------------------------------------- numeric helpers

const ema = (previous, value, weight) => (previous === null ? value : weight * value + (1 - weight) * previous);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const below = Math.floor(position);
  return sorted[below] + (sorted[Math.min(below + 1, sorted.length - 1)] - sorted[below]) * (position - below);
}

function variance(values) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
}

/**
 * Dominant period (in samples) from the normalized autocorrelation, or null when no lag in
 * [minLag, n/2] is convincing. Among strong peaks the shortest lag wins, so a period is never
 * mistaken for twice itself.
 */
export function estimatePeriod(values, minLag, minConfidence) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const centered = values.map((v) => v - mean);
  const energy = centered.reduce((sum, v) => sum + v * v, 0) / n;
  if (energy === 0) return null;

  const r = [];
  for (let lag = 0; lag <= Math.floor(n / 2); lag += 1) {
    let sum = 0;
    for (let i = 0; i + lag < n; i += 1) sum += centered[i] * centered[i + lag];
    r.push(sum / (n - lag) / energy);
  }
  const peaks = [];
  for (let lag = Math.max(minLag, 1); lag < r.length - 1; lag += 1) {
    if (r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1] && r[lag] >= minConfidence) peaks.push(lag);
  }
  if (!peaks.length) return null;
  const best = Math.max(...peaks.map((lag) => r[lag]));
  return peaks.find((lag) => r[lag] >= 0.9 * best);
}

// ---------------------------------------------------------------- signal-level counter

/** Counts reps on a scalar signal sampled at a fixed rate. Exported so it can be tested alone. */
export function createSignalCounter(options = {}) {
  const config = { ...DEFAULTS, ...options };
  return Object.freeze({
    config,
    capacity: Math.max(4, Math.round(config.bufferSeconds * config.sampleFps)),
    buffer: Object.freeze([]),
    smoothed: null,
    home: null,
    homeRange: 0,       // the range when home was last chosen …
    homeCount: 0,       // … and the count then; a rep counted since confirms the choice
    zone: null,
    awayDepth: 0,       // how far the current swing has gone across the range, 0 = not away yet
    periodSeconds: null,
    lastRepMs: -Infinity,
    count: 0,
    repTimes: Object.freeze([]),
  });
}

const minAmplitudeOf = (config) => (typeof config.minAmplitude === 'number'
  ? config.minAmplitude : config.minAmplitude[config.signal]);

/** Forget which side is home (the movement changed); the count and its timing stay. */
export function resetZones(state) {
  return Object.freeze({ ...state, home: null, homeRange: 0, zone: null, awayDepth: 0 });
}

/** One rep = home → away → home; the gap only holds back swings that stayed shallow. */
function transition(state, zone, awayness, timeMs) {
  const { config } = state;
  if (zone === null) return {};
  if (zone !== state.home) return { zone, awayDepth: Math.max(state.awayDepth, awayness) };
  if (state.zone === state.home || state.awayDepth === 0) return { zone, awayDepth: 0 };

  const sinceLast = timeMs - state.lastRepMs;
  const gapSeconds = state.awayDepth >= config.fullSwing || !state.periodSeconds
    ? config.fallbackGapSeconds : config.gapFraction * state.periodSeconds;
  if (sinceLast < 1000 * Math.max(gapSeconds, config.fallbackGapSeconds)) return { zone, awayDepth: 0 };
  return { zone, awayDepth: 0, count: state.count + 1,
    repTimes: Object.freeze([...state.repTimes, timeMs]), lastRepMs: timeMs };
}

/**
 * Advance on a whole window of one consistent signal; its last value is the current sample.
 * @param {number[]} window signal history, oldest first
 * @param {number} timeMs time of the last sample; a counted rep is reported at this time
 */
export function stepWindow(state, window, timeMs) {
  const { config } = state;
  const low = quantile(window, 0.05);
  const range = quantile(window, 0.95) - low;
  if (!(range >= minAmplitudeOf(config))) return state;

  const lag = estimatePeriod(window, Math.round(config.minPeriodSeconds * config.sampleFps), config.minPeriodConfidence);
  const next = { ...state, ...placeHome(state, window, low, range), periodSeconds: lag ? lag / config.sampleFps : null };
  const current = window[window.length - 1];
  const position = (current - low) / range;
  const awayness = next.home === ZONE.HIGH ? 1 - position : position;
  let { zone } = next;
  if (current >= low + config.highZone * range) zone = ZONE.HIGH;
  else if (current <= low + config.lowZone * range) zone = ZONE.LOW;
  return Object.freeze({ ...next, ...transition(next, zone, awayness, timeMs) });
}

/**
 * "Home" is the side the window started on: the resting posture, read from a half-second mean so one
 * noisy sample cannot land it on the wrong side. Until a rep has been counted under it, it is read
 * again whenever the range more than doubles: a home chosen while only tracking jitter was moving must
 * not bind the real movement that follows. Once a rep confirms it, it stays — by then the head of the
 * window is mid-set, not the resting posture. Zones restart only if home changes; the count never does.
 */
function placeHome(state, window, low, range) {
  const { config } = state;
  const confirmed = state.count > state.homeCount;
  if (state.home !== null && (confirmed || range <= config.homeRegrowth * state.homeRange)) return {};
  const head = window.slice(0, Math.max(1, Math.round(config.homeSeconds * config.sampleFps)));
  const start = head.reduce((a, b) => a + b, 0) / head.length;
  const home = start >= low + range / 2 ? ZONE.HIGH : ZONE.LOW;
  const chosen = { homeRange: range, homeCount: state.count };
  return home === state.home ? chosen : { ...chosen, home, zone: null, awayDepth: 0 };
}

/** Sample-by-sample entry point: smooths, keeps its own window, then calls stepWindow. */
export function updateSignalCounter(state, value, timeMs) {
  if (!Number.isFinite(value)) return state;
  const smoothed = ema(state.smoothed, value, state.config.smoothing);
  const buffer = Object.freeze([...state.buffer, smoothed].slice(-state.capacity));
  return stepWindow(Object.freeze({ ...state, buffer, smoothed }), buffer, timeMs);
}

// ---------------------------------------------------------------- choosing the signal

/** Leading eigenvector of the window's covariance, warm-started and sign-aligned with `previous`. */
export function principalComponent(window, previous, iterations = DEFAULTS.powerIterations) {
  const dims = window[0].length;
  const mean = columnMeans(window);
  const centered = window.map((row) => row.map((v, d) => v - mean[d]));

  let vector = previous ? Float64Array.from(previous) : new Float64Array(dims).fill(1 / Math.sqrt(dims));
  for (let it = 0; it < iterations; it += 1) {
    const next = new Float64Array(dims);
    centered.forEach((row) => {
      const projection = dot(row, vector);
      row.forEach((v, d) => { next[d] += projection * v; });
    });
    const norm = Math.hypot(...next);
    if (norm === 0) return previous ?? vector;
    vector = next.map((v) => v / norm);
  }
  // The sign of an eigenvector is arbitrary; flipping between samples would invert the signal.
  return previous && dot(vector, previous) < 0 ? vector.map((v) => -v) : vector;
}

function columnMeans(window) {
  const mean = new Float64Array(window[0].length);
  window.forEach((row) => row.forEach((v, d) => { mean[d] += v / window.length; }));
  return mean;
}

/** Index of the angle with the largest variance, switching only when another clearly wins. */
export function busiestAngle(window, current, switchRatio = DEFAULTS.channelSwitchRatio) {
  const variances = Array.from({ length: 8 }, (_, a) => variance(window.map((row) => row[ANGLE_OFFSET + a])));
  const best = variances.indexOf(Math.max(...variances));
  if (current === null || current === undefined) return best;
  return variances[best] > switchRatio * variances[current] ? best : current;
}

/** The signal over the whole window under the current definition, plus that definition. */
function readSignal(state, window) {
  const { config } = state;
  if (config.signal === SIGNAL.ANGLE) {
    const channel = busiestAngle(window, state.channel, config.channelSwitchRatio);
    const changed = state.channel !== null && channel !== state.channel;
    return { channel, changed, series: window.map((row) => row[ANGLE_OFFSET + channel]) };
  }
  if (window.length < 2) return { component: state.component, changed: false, series: [0] };
  const component = principalComponent(window, state.component, config.powerIterations);
  const mean = columnMeans(window);
  const changed = state.component !== null && dot(component, state.component) < config.axisResetDot;
  return { component, changed, series: window.map((row) => dot(row, component) - dot(mean, component)) };
}

// ---------------------------------------------------------------- frame-level counter

export function createGenericCounter(options = {}) {
  const config = { ...DEFAULTS, ...options };
  return Object.freeze({
    config,
    signalCounter: createSignalCounter(config),
    nextSampleMs: null,
    motion: null,
    smoothedMotion: null,
    window: Object.freeze([]),
    component: null,
    channel: null,
    signal: null,
    count: 0,
    repTimes: Object.freeze([]),
  });
}

function motionVector(frame, size) {
  const features = frameFeatures(normalizePoints(selectLandmarks(frame, size.width, size.height)));
  const motion = Float64Array.from(MOTION_COLUMNS, (column) => features[column]);
  return motion.every(Number.isFinite) ? motion : null;
}

/** One resampled tick: smooth, extend the window, re-read the signal, advance the counter. */
function sampleTick(state, timeMs) {
  const weight = state.config.smoothing;
  const smoothedMotion = state.smoothedMotion
    ? state.motion.map((v, d) => weight * v + (1 - weight) * state.smoothedMotion[d]) : state.motion;
  const window = Object.freeze([...state.window, smoothedMotion].slice(-state.signalCounter.capacity));
  const { series, changed, component = state.component, channel = state.channel } = readSignal(state, window);
  const base = changed ? resetZones(state.signalCounter) : state.signalCounter;
  const signalCounter = stepWindow(base, series, timeMs);
  return { ...state, smoothedMotion, window, component, channel, signal: series[series.length - 1], signalCounter,
    count: signalCounter.count, repTimes: signalCounter.repTimes };
}

/**
 * Fold one video frame in.
 * @param {Array<{x: number, y: number, visibility?: number}>|null|undefined} frame MediaPipe's 33
 *   landmarks, or null/undefined when no pose was found (the last pose is then held)
 * @param {number} timestampMs frame time; must not decrease
 * @param {{ width: number, height: number }} size source frame size, for the aspect correction
 */
export function updateGenericCounter(state, frame, timestampMs, size) {
  const motion = motionVector(frame, size) ?? state.motion;
  if (!motion) return state;
  let next = { ...state, motion, nextSampleMs: state.nextSampleMs ?? timestampMs };

  const stepMs = 1000 / state.config.sampleFps;
  let ticks = 0;
  // A jump (seek, dropped frames) emits at most one window of held samples, never an unbounded loop.
  while (timestampMs >= next.nextSampleMs && ticks < state.signalCounter.capacity) {
    next = { ...sampleTick(next, next.nextSampleMs), nextSampleMs: next.nextSampleMs + stepMs };
    ticks += 1;
  }
  if (timestampMs >= next.nextSampleMs) next = { ...next, nextSampleMs: timestampMs + stepMs };
  return Object.freeze(next);
}
