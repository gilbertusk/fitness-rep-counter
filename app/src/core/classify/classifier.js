/**
 * Post-processing for the exercise classifier: raw model scores → a stable label.
 *
 * Pure and immutable — the ONNX session itself lives in `app/src/adapters/onnxClassifier.js`.
 * One window arrives every WINDOW_STRIDE frames (1 second at 15 fps), so a bare argmax flickers
 * between similar classes; probabilities are smoothed with an EMA across windows before the
 * winner is read off.
 */

/**
 * Provisional thresholds — the same status as the rep-counter angles in `core/exercises.js`.
 * Stage 1 replaces them with the values swept on the validation set, written into
 * `app/models/labels.json` by `python -m repcount.evaluation.classifier_report` and passed to
 * `createClassifier`. Until a model exists these defaults are only placeholders, not measurements.
 */
export const DEFAULT_THRESHOLDS = Object.freeze({
  smoothing: 0.4,      // EMA weight of the newest window
  minConfidence: 0.5,  // top probability below this → "tidak yakin"
  minMargin: 0.15,     // top-1 minus top-2 below this → "tidak yakin"
});

export const UNCERTAIN = Object.freeze({
  LOW_CONFIDENCE: 'low-confidence',
  CLOSE_CALL: 'close-call',
  POOR_POSE: 'poor-pose',
});

/** Numerically stable softmax; shifting by the maximum keeps exp() from overflowing. */
export function softmax(scores) {
  const highest = Math.max(...scores);
  const exponentials = Array.from(scores, (score) => Math.exp(score - highest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

/** Indices of the two highest probabilities, best first. */
function topTwo(probabilities) {
  return probabilities
    .map((probability, index) => ({ probability, index }))
    .sort((a, b) => b.probability - a.probability || a.index - b.index)
    .slice(0, 2);
}

/**
 * @param {{ labels: string[], thresholds?: object }} options
 * @returns {{ labels: string[], thresholds: object, probabilities: number[]|null, nWindows: number }}
 */
export function createClassifier({ labels, thresholds = {} }) {
  if (!labels?.length) throw new Error('createClassifier needs a non-empty labels array');
  return {
    labels,
    thresholds: { ...DEFAULT_THRESHOLDS, ...thresholds },
    probabilities: null,
    nWindows: 0,
  };
}

/** Exponential moving average across windows; the first window seeds the average. */
function smooth(previous, current, weight) {
  if (!previous) return current;
  return current.map((value, i) => weight * value + (1 - weight) * previous[i]);
}

function verdict(probabilities, labels, thresholds, poorPose) {
  const [best, runnerUp] = topTwo(probabilities);
  const margin = best.probability - (runnerUp?.probability ?? 0);

  let reason = null;
  if (poorPose) reason = UNCERTAIN.POOR_POSE;
  else if (best.probability < thresholds.minConfidence) reason = UNCERTAIN.LOW_CONFIDENCE;
  else if (margin < thresholds.minMargin) reason = UNCERTAIN.CLOSE_CALL;

  return {
    label: labels[best.index],
    confidence: best.probability,
    margin,
    uncertain: reason !== null,
    uncertainReason: reason,
  };
}

/**
 * Fold one window's model output into the classifier state.
 * @param {object} state from `createClassifier`
 * @param {number[]} scores one logit per class, straight from the model
 * @param {{ poorPose?: boolean }} context `poorPose` marks a window over the missing-frame limit
 *   (docs/FEATURES.md §8); its prediction still updates the average but is reported as uncertain
 * @returns {object} the next state, with `prediction` describing the current best guess
 */
export function updateClassifier(state, scores, { poorPose = false } = {}) {
  if (scores.length !== state.labels.length) {
    throw new Error(`model returned ${scores.length} scores for ${state.labels.length} labels`);
  }
  const probabilities = smooth(state.probabilities, softmax(scores), state.thresholds.smoothing);
  return {
    ...state,
    probabilities,
    nWindows: state.nWindows + 1,
    prediction: verdict(probabilities, state.labels, state.thresholds, poorPose),
  };
}

/** Forget the smoothed history, e.g. when the user switches video or restarts a session. */
export function resetClassifier(state) {
  return { ...state, probabilities: null, nWindows: 0, prediction: undefined };
}
