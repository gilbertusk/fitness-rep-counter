/**
 * Plank timer: runs while the body is in position, survives a short tracking dropout, and remembers
 * the best hold. "In position" is decided by the plank form rule (core/form/rules/plank.js), not here.
 * Pure and immutable.
 */

export const HOLD_DEFAULTS = Object.freeze({
  graceMs: 400,     // a dropout shorter than this does not end the hold
  minHoldMs: 1000,  // shorter holds are not recorded as holds
});

export function createHoldTimer(options = {}) {
  return Object.freeze({
    config: { ...HOLD_DEFAULTS, ...options },
    holding: false,
    startMs: null,
    lastGoodMs: null,
    currentMs: 0,
    bestMs: 0,
    holds: Object.freeze([]),
  });
}

function endHold(state) {
  const duration = state.lastGoodMs - state.startMs;
  const holds = duration >= state.config.minHoldMs ? Object.freeze([...state.holds, duration]) : state.holds;
  return Object.freeze({ ...state, holding: false, startMs: null, currentMs: 0, holds });
}

/**
 * @param {boolean} inPosition whether this frame's pose satisfies the plank rule
 * @param {number} timeMs frame time
 */
export function updateHoldTimer(state, inPosition, timeMs) {
  if (inPosition) {
    const startMs = state.holding ? state.startMs : timeMs;
    const currentMs = timeMs - startMs;
    return Object.freeze({ ...state, holding: true, startMs, lastGoodMs: timeMs, currentMs,
      bestMs: Math.max(state.bestMs, currentMs) });
  }
  if (!state.holding) return state;
  if (timeMs - state.lastGoodMs > state.config.graceMs) return endHold(state);
  return Object.freeze({ ...state, currentMs: timeMs - state.startMs });
}

/** Close a running hold, e.g. when the set ends or the user switches exercise. */
export function finishHold(state) {
  return state.holding ? endHold(state) : state;
}
