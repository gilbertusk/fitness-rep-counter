const MEDIAPIPE_VERSION = '1.0.1';
const ONNXRUNTIME_VERSION = '1.23.2';

export const MEDIAPIPE_BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
export const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
// Must stay identical to POSE_MODEL_URL in ml/src/repcount/config.py (train/serve consistency).
export const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

// WASM-only bundle: the WebGPU one pulls a 25.5 MB runtime instead of 11.9 MB, for a 64k-parameter model
// that runs in about 1 ms on the CPU (reports/05-performance/performance.md).
export const ONNXRUNTIME_BUNDLE_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNXRUNTIME_VERSION}/dist/ort.wasm.min.mjs`;
export const ONNXRUNTIME_WASM_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNXRUNTIME_VERSION}/dist/`;
// Written by `python -m repcount.export.onnx`; served from app/models/ alongside index.html.
export const CLASSIFIER_MODEL_URL = './models/exercise_classifier.onnx';
export const CLASSIFIER_LABELS_URL = './models/labels.json';
