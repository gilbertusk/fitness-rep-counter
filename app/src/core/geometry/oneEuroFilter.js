/**
 * One Euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter whose cutoff rises with
 * speed, so a still pose loses its jitter while a fast movement keeps up without lag.
 *
 * Used on landmarks for what the user *sees* and for the form rules and plank timer — never on the
 * classifier features or the rep counter, which must see the same raw keypoints they were trained and
 * evaluated on (docs/PLAN.md §3; the deviation from the stage 4 prompt is recorded in §8).
 *
 * Pure and immutable.
 */

// Starting values for MediaPipe's normalized [0, 1] coordinates: at rest (≈ 0.02 units/s) the cutoff
// stays near 1 Hz and jitter is removed; a limb moving 1 unit/s lifts it to ≈ 11 Hz.
export const ONE_EURO_DEFAULTS = Object.freeze({ minCutoff: 1.0, beta: 10, dCutoff: 1.0 });

const alpha = (cutoffHz, dtSeconds) => {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dtSeconds);
};

/** One scalar step. `state` is null before the first sample. Returns { state, value }. */
export function filterValue(state, x, timeMs, options = ONE_EURO_DEFAULTS) {
  if (!Number.isFinite(x)) return { state, value: x };
  if (!state || timeMs <= state.timeMs) return { state: Object.freeze({ x, dx: 0, timeMs }), value: x };

  const { minCutoff, beta, dCutoff } = { ...ONE_EURO_DEFAULTS, ...options };
  const dt = (timeMs - state.timeMs) / 1000;
  const rawSpeed = (x - state.x) / dt;
  const dx = state.dx + alpha(dCutoff, dt) * (rawSpeed - state.dx);
  const value = state.x + alpha(minCutoff + beta * Math.abs(dx), dt) * (x - state.x);
  return { state: Object.freeze({ x: value, dx, timeMs }), value };
}

/**
 * Filter every landmark's x, y and z; visibility passes through untouched.
 * @param {Array|null} state per-landmark filter states from the previous call, or null
 * @param {Array<{x: number, y: number, z: number, visibility?: number}>|null|undefined} landmarks
 * @returns {{ state: Array|null, landmarks: Array|null }}
 */
export function filterPose(state, landmarks, timeMs, options = ONE_EURO_DEFAULTS) {
  if (!landmarks) return { state, landmarks: null };
  const states = [];
  const filtered = landmarks.map((point, i) => {
    const previous = state?.[i] ?? {};
    const x = filterValue(previous.x ?? null, point.x, timeMs, options);
    const y = filterValue(previous.y ?? null, point.y, timeMs, options);
    const z = filterValue(previous.z ?? null, point.z, timeMs, options);
    states.push(Object.freeze({ x: x.state, y: y.state, z: z.state }));
    return { ...point, x: x.value, y: y.value, z: z.value };
  });
  return { state: Object.freeze(states), landmarks: filtered };
}
