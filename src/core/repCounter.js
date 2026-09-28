export const PHASE = Object.freeze({ UP: 'up', DOWN: 'down' });

export const FEEDBACK = Object.freeze({
  GOOD: 'Bagus!',
  PARTIAL: 'Kurang dalam, turun lagi',
});

export function createCounter() {
  return Object.freeze({ phase: PHASE.UP, count: 0, lowestAngle: Infinity, feedback: null });
}

/**
 * Pure state machine: returns the next counter state for a new joint angle.
 * Uses two thresholds (hysteresis) so jitter near one threshold cannot double count.
 *   UP --(angle <= downAngle)--> DOWN --(angle >= upAngle)--> UP, count + 1
 * A dip below partialAngle that never reaches downAngle is reported as a partial rep.
 */
export function updateCounter(state, angle, { downAngle, upAngle, partialAngle }) {
  if (typeof angle !== 'number' || Number.isNaN(angle)) return state;

  const lowestAngle = Math.min(state.lowestAngle, angle);

  if (state.phase === PHASE.UP && angle <= downAngle) {
    return Object.freeze({ ...state, phase: PHASE.DOWN, lowestAngle, feedback: null });
  }

  if (state.phase === PHASE.DOWN && angle >= upAngle) {
    return Object.freeze({
      phase: PHASE.UP,
      count: state.count + 1,
      lowestAngle: Infinity,
      feedback: FEEDBACK.GOOD,
    });
  }

  if (state.phase === PHASE.UP && angle >= upAngle) {
    const isPartial = lowestAngle <= partialAngle;
    return Object.freeze({
      ...state,
      lowestAngle: Infinity,
      feedback: isPartial ? FEEDBACK.PARTIAL : state.feedback,
    });
  }

  return Object.freeze({ ...state, lowestAngle });
}
