import { createGenericCounter, updateGenericCounter, SIGNAL } from '../../../app/src/core/counting/genericCounter.js';
import { createNaivePeakCounter, updateNaivePeakCounter } from '../../../app/src/core/counting/naivePeakCounter.js';
import { createCounter, updateCounter } from '../../../app/src/core/counting/thresholdCounter.js';
import { measureExercise } from '../../../app/src/core/form/measure.js';
import { EXERCISES } from '../../../app/src/core/exercises.js';

/**
 * Name → counter, all behind one interface so the harness can stream any of them:
 *   supports(label)        false → the video is reported N/A for this counter, never as zero
 *   create(label, params)  a fresh state for one video
 *   update(state, frame, timestampMs, size)
 *   result(state)          { count, repTimes }
 */

// The threshold counter only has configurations for these dataset labels.
const THRESHOLD_EXERCISES = Object.freeze({ squat: EXERCISES.squat, push_up: EXERCISES.pushup });

const generic = (signal) => ({
  description: `genericCounter.js, signal = ${signal}`,
  supports: () => true,
  create: (_label, params = {}) => createGenericCounter({ ...params, signal }),
  update: updateGenericCounter,
  result: ({ count, repTimes }) => ({ count, repTimes: [...repTimes] }),
});

export const COUNTERS = Object.freeze({
  generic: generic(SIGNAL.PCA),
  'generic-angle': generic(SIGNAL.ANGLE),
  'naive-peaks': {
    description: 'naivePeakCounter.js: peaks of the busiest angle, fixed parameters (lower bound)',
    supports: () => true,
    create: (_label, params = {}) => createNaivePeakCounter(params),
    update: updateNaivePeakCounter,
    result: ({ count, repTimes }) => ({ count, repTimes: [...repTimes] }),
  },
  threshold: {
    description: 'thresholdCounter.js with the hand-set angles in core/exercises.js (squat, push-up only)',
    supports: (label) => label in THRESHOLD_EXERCISES,
    create: (label) => ({ exercise: THRESHOLD_EXERCISES[label], counter: createCounter(), repTimes: [] }),
    update: (state, frame, timestampMs, size) => {
      const { angle } = measureExercise(frame, state.exercise, size);
      const counter = updateCounter(state.counter, angle, state.exercise);
      const repTimes = counter.count > state.counter.count ? [...state.repTimes, timestampMs] : state.repTimes;
      return { ...state, counter, repTimes };
    },
    result: (state) => ({ count: state.counter.count, repTimes: [...state.repTimes] }),
  },
});

// JSON has no NaN, so a missing value arrives as null — and in JavaScript null * 2 is 0, which would
// read a missing landmark as a point at the image corner. Turn it back into NaN.
const value = (v) => (v === null || v === undefined ? NaN : v);

/** One frame of data/keypoints_json → MediaPipe-shaped landmarks, or null when no pose was found. */
export function toLandmarks(frame) {
  if (!frame) return null;
  return frame.map(([x, y, z, visibility]) => ({ x: value(x), y: value(y), z: value(z), visibility: value(visibility) }));
}

/**
 * Stream one video through a counter, frame by frame, as the browser would.
 * @param {{ label: string, width: number, height: number, timestamps_ms: number[], landmarks: Array }} video
 * @returns {{ count: number, repTimes: number[], msPerFrame: number } | null} null when unsupported
 */
export function runCounter(name, video, params = {}) {
  const counter = COUNTERS[name];
  if (!counter) throw new Error(`unknown counter ${name}; known: ${Object.keys(COUNTERS).join(', ')}`);
  if (!counter.supports(video.label)) return null;

  const size = { width: video.width, height: video.height };
  let state = counter.create(video.label, params);
  const started = performance.now();
  video.landmarks.forEach((frame, i) => {
    state = counter.update(state, toLandmarks(frame), video.timestamps_ms[i], size);
  });
  const elapsed = performance.now() - started;
  return { ...counter.result(state), msPerFrame: video.landmarks.length ? elapsed / video.landmarks.length : 0 };
}
