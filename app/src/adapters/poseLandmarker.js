import { MEDIAPIPE_BUNDLE_URL, MEDIAPIPE_WASM_URL, POSE_MODEL_URL } from '../config.js';

/**
 * Loads MediaPipe Pose Landmarker (VIDEO mode, one pose).
 * @returns {Promise<{ vision: object, detect: (video: HTMLVideoElement, timestampMs: number) => object[]|undefined }>}
 *   `detect` returns the 33 normalized landmarks of the first pose, or undefined when none is found.
 */
export async function createPoseLandmarker() {
  const vision = await import(MEDIAPIPE_BUNDLE_URL);
  const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
  const landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: 1,
  });

  const detect = (video, timestampMs) => landmarker.detectForVideo(video, timestampMs).landmarks[0];
  return { vision, detect };
}
