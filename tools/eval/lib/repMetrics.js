/**
 * Metrics for comparing a rep counter with human labels (labels/README.md). Pure functions only.
 *
 * Count metrics are averaged per video (MAE, OBO, normalized MAE): every video weighs the same,
 * however many reps it holds. Per-rep metrics (precision, recall, latency) are pooled over all reps,
 * so a 20-rep video counts twenty times as much as a 1-rep one there.
 */

export const MATCH_TOLERANCE_MS = 1000;

const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);

/**
 * Pair predicted rep times with labelled ones, one-to-one, closest pairs first, within tolerance.
 * @param {number[]} predicted ms
 * @param {number[]} truth ms
 * @returns {{ pairs: {predicted: number, truth: number}[], unmatchedPredicted: number[], unmatchedTruth: number[] }}
 */
export function matchReps(predicted, truth, toleranceMs = MATCH_TOLERANCE_MS) {
  const candidates = [];
  predicted.forEach((p, i) => truth.forEach((t, j) => {
    if (Math.abs(p - t) <= toleranceMs) candidates.push({ i, j, distance: Math.abs(p - t) });
  }));
  candidates.sort((a, b) => a.distance - b.distance || a.j - b.j || a.i - b.i);

  const usedP = new Set();
  const usedT = new Set();
  const pairs = [];
  for (const { i, j } of candidates) {
    if (usedP.has(i) || usedT.has(j)) continue;
    usedP.add(i);
    usedT.add(j);
    pairs.push({ predicted: predicted[i], truth: truth[j] });
  }
  pairs.sort((a, b) => a.truth - b.truth);
  return {
    pairs,
    unmatchedPredicted: predicted.filter((_, i) => !usedP.has(i)),
    unmatchedTruth: truth.filter((_, j) => !usedT.has(j)),
  };
}

/**
 * Metrics for one video.
 * @param {{ predicted: number[], truth: number[] }} reps rep times in ms
 */
export function videoMetrics({ predicted, truth }, toleranceMs = MATCH_TOLERANCE_MS) {
  const { pairs } = matchReps(predicted, truth, toleranceMs);
  const error = predicted.length - truth.length;
  return {
    predictedCount: predicted.length,
    truthCount: truth.length,
    error,
    absError: Math.abs(error),
    offByOne: Math.abs(error) <= 1,
    normalizedError: truth.length ? Math.abs(error) / truth.length : null,
    matched: pairs.length,
    precision: predicted.length ? pairs.length / predicted.length : null,
    recall: truth.length ? pairs.length / truth.length : null,
    // Positive = the counter reports the rep after the human mark (it has to see the rep end first).
    latencyMs: mean(pairs.map((pair) => pair.predicted - pair.truth)),
    latencies: pairs.map((pair) => pair.predicted - pair.truth),
  };
}

/** Summary over many videos (rows from videoMetrics). Empty input gives n = 0 and null metrics. */
export function aggregate(rows) {
  const predicted = rows.reduce((sum, r) => sum + r.predictedCount, 0);
  const truth = rows.reduce((sum, r) => sum + r.truthCount, 0);
  const matched = rows.reduce((sum, r) => sum + r.matched, 0);
  return {
    n: rows.length,
    mae: mean(rows.map((r) => r.absError)),
    obo: mean(rows.map((r) => (r.offByOne ? 1 : 0))),
    normalizedMae: mean(rows.filter((r) => r.normalizedError !== null).map((r) => r.normalizedError)),
    precision: predicted ? matched / predicted : null,
    recall: truth ? matched / truth : null,
    latencyMs: mean(rows.flatMap((r) => r.latencies)),
  };
}

/** aggregate() per class, keyed by label, in label order. */
export function aggregateByClass(rows) {
  const labels = [...new Set(rows.map((r) => r.label))].sort();
  return Object.fromEntries(labels.map((label) => [label, aggregate(rows.filter((r) => r.label === label))]));
}

/**
 * The headline summary labels/README.md promises: every video, and again without the ambiguous ones.
 * Rows a counter cannot handle (status 'n/a', e.g. the threshold counter on a class it has no
 * configuration for) are left out and counted separately, so they never read as zero errors.
 */
export function summarize(rows) {
  const scored = rows.filter((r) => r.status !== 'n/a');
  const clear = scored.filter((r) => !r.ambiguous);
  return {
    all: aggregate(scored),
    withoutAmbiguous: aggregate(clear),
    byClass: aggregateByClass(scored),
    notApplicable: rows.length - scored.length,
  };
}
