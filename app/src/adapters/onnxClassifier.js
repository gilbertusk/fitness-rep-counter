import {
  CLASSIFIER_LABELS_URL, CLASSIFIER_MODEL_URL, ONNXRUNTIME_BUNDLE_URL, ONNXRUNTIME_WASM_URL,
} from '../config.js';
import { N_FEATURES, WINDOW_FRAMES } from '../core/features/features.js';

/**
 * Loads the exercise classifier and runs one window through it.
 *
 * Adapter only: no post-processing lives here. The scores it returns go to
 * `app/src/core/classify/classifier.js`, which is where smoothing and the
 * "tidak yakin" thresholds are decided.
 */

/** @param {Float64Array[]} window WINDOW_FRAMES rows of N_FEATURES values (docs/FEATURES.md §8) */
function toTensor(ort, window) {
  if (window.length !== WINDOW_FRAMES) {
    throw new Error(`classifier expects ${WINDOW_FRAMES} frames, got ${window.length}`);
  }
  const flat = new Float32Array(WINDOW_FRAMES * N_FEATURES);
  window.forEach((row, frame) => flat.set(row, frame * N_FEATURES));
  return new ort.Tensor('float32', flat, [1, WINDOW_FRAMES, N_FEATURES]);
}

/**
 * @returns {Promise<{ labels: string[], thresholds: object, classify: (window: Float64Array[]) => Promise<number[]> }>}
 *   `classify` resolves to one score per label, in the order of `labels`
 */
export async function createExerciseClassifier({
  modelUrl = CLASSIFIER_MODEL_URL,
  labelsUrl = CLASSIFIER_LABELS_URL,
} = {}) {
  const ort = await import(ONNXRUNTIME_BUNDLE_URL);
  ort.env.wasm.wasmPaths = ONNXRUNTIME_WASM_URL;

  const [session, metadata] = await Promise.all([
    ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu', 'wasm'] }),
    fetch(labelsUrl).then((response) => {
      if (!response.ok) throw new Error(`cannot load ${labelsUrl}: ${response.status}`);
      return response.json();
    }),
  ]);

  const [inputName] = session.inputNames;
  const [outputName] = session.outputNames;

  const classify = async (window) => {
    const output = await session.run({ [inputName]: toTensor(ort, window) });
    return Array.from(output[outputName].data);
  };

  return { labels: metadata.labels, thresholds: metadata.thresholds ?? {}, classify };
}
