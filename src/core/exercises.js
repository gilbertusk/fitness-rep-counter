// MediaPipe Pose landmark indices: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
export const LANDMARK = Object.freeze({
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
});

const L = LANDMARK;

/**
 * Each exercise defines, per body side:
 *  - joint:   [a, b, c] → angle at b drives the rep counter
 *  - torso / bodyLine: points used by the form rules
 * Thresholds are in degrees and are starting values, tune them with real videos.
 */
export const EXERCISES = Object.freeze({
  squat: Object.freeze({
    id: 'squat',
    label: 'Squat',
    sides: {
      left: { joint: [L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE], torso: [L.LEFT_SHOULDER, L.LEFT_HIP] },
      right: { joint: [L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE], torso: [L.RIGHT_SHOULDER, L.RIGHT_HIP] },
    },
    downAngle: 95,
    upAngle: 160,
    partialAngle: 130,
    maxTorsoLean: 45,
  }),
  pushup: Object.freeze({
    id: 'pushup',
    label: 'Push-up',
    sides: {
      left: { joint: [L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST], bodyLine: [L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_ANKLE] },
      right: { joint: [L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST], bodyLine: [L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_ANKLE] },
    },
    downAngle: 90,
    upAngle: 155,
    partialAngle: 125,
    minBodyLine: 160,
  }),
});
