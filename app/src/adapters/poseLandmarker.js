import { MEDIAPIPE_BUNDLE_URL, MEDIAPIPE_WASM_URL, POSE_MODEL_URL } from '../config.js';

/**
 * Software WebGL (SwiftShader, llvmpipe, …) runs MediaPipe's GPU path on the CPU through an emulator,
 * and far slower than its real CPU path: 245 ms vs 35 ms per frame, measured in headless Chromium on a
 * 4-core machine. VMs, remote desktops and Linux without GPU drivers fall in the same trap.
 */
export function prefersCpu(renderer) {
  return !renderer || /swiftshader|llvmpipe|softpipe|software|basic render/i.test(renderer);
}

function webglRenderer() {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    if (!gl) return null;
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    return String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  } catch {
    return null;
  }
}

/**
 * Loads MediaPipe Pose Landmarker (VIDEO mode, one pose) on the GPU when there is a real one, on the
 * CPU otherwise, and on the other one if the first choice fails to start.
 * @returns {Promise<{ vision: object, delegate: string, renderer: string|null, detect: (video: HTMLVideoElement, timestampMs: number) => object[]|undefined }>}
 *   `detect` returns the 33 normalized landmarks of the first pose, or undefined when none is found.
 */
export async function createPoseLandmarker() {
  const vision = await import(MEDIAPIPE_BUNDLE_URL);
  const fileset = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
  const create = (delegate) => vision.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
  });

  const renderer = webglRenderer();
  let delegate = prefersCpu(renderer) ? 'CPU' : 'GPU';
  let landmarker;
  try {
    landmarker = await create(delegate);
  } catch {
    delegate = delegate === 'GPU' ? 'CPU' : 'GPU';
    landmarker = await create(delegate);
  }

  const detect = (video, timestampMs) => landmarker.detectForVideo(video, timestampMs).landmarks[0];
  return { vision, delegate, renderer, detect };
}
