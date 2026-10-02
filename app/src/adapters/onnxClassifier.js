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
  // labels.json first: until stage 1 exports a model it lists no labels, and then neither the runtime
  // nor the model is fetched — no wasted download, no 404s in the console.
  const response = await fetch(labelsUrl);
  if (!response.ok) throw new Error(`cannot load ${labelsUrl}: ${response.status}`);
  const metadata = await response.json();
  if (!metadata.labels?.length) throw new Error('no trained exercise classifier yet');

  const ort = await import(ONNXRUNTIME_BUNDLE_URL);
  ort.env.wasm.wasmPaths = ONNXRUNTIME_WASM_URL;
  // Warnings (e.g. "Unknown CPU vendor" in VMs) go to console.error and mean nothing to the user.
  ort.env.logLevel = 'error';
  const session = await ort.InferenceSession.create(modelUrl, { executionProviders: ['webgpu', 'wasm'] });

  const [inputName] = session.inputNames;
  const [outputName] = session.outputNames;

  const classify = async (window) => {
    const output = await session.run({ [inputName]: toTensor(ort, window) });
    return Array.from(output[outputName].data);
  };

  return { labels: metadata.labels, thresholds: metadata.thresholds ?? {}, classify };
}
