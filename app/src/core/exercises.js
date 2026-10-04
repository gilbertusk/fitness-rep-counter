// MediaPipe Pose landmark indices: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
export const LANDMARK = Object.freeze({
  NOSE: 0,
  LEFT_EAR: 7, RIGHT_EAR: 8,
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

/**
 * Status of each exercise in the app. Nothing has been evaluated yet (docs/PLAN.md §8: stage 1 is
 * untrained and stage 3 has no labels), so every exercise is shown as experimental. Once
 * reports/03-rep-counter/rep_counter.md exists, exercises that hold up move to SUPPORTED there and here.
 */
export const STATUS = Object.freeze({ SUPPORTED: 'supported', EXPERIMENTAL: 'experimental' });

const entry = (name, extra = {}) => Object.freeze({ name, task: 'reps', status: STATUS.EXPERIMENTAL, ...extra });

/**
 * The 22 dataset classes (snake_case labels from ml/splits/split_v1.json), in label order.
 * `note` carries a risk documented in docs/PLAN.md §7 — a known limitation, not a measurement.
 */
export const CATALOG = Object.freeze({
  barbell_biceps_curl: entry('Barbell biceps curl'),
  bench_press: entry('Bench press', { note: 'Posisi berbaring: deteksi pose lebih lemah' }),
  chest_fly_machine: entry('Chest fly (mesin)'),
  deadlift: entry('Deadlift'),
  decline_bench_press: entry('Decline bench press', { note: 'Data sedikit dan deteksi pose terlemah' }),
  hammer_curl: entry('Hammer curl'),
  hip_thrust: entry('Hip thrust'),
  incline_bench_press: entry('Incline bench press'),
  lat_pulldown: entry('Lat pulldown'),
  lateral_raise: entry('Lateral raise'),
  leg_extension: entry('Leg extension'),
  leg_raises: entry('Leg raise'),
  plank: entry('Plank', { task: 'hold', note: 'Hanya 7 video di dataset' }),
  pull_up: entry('Pull-up'),
  push_up: entry('Push-up'),
  romanian_deadlift: entry('Romanian deadlift'),
  russian_twist: entry('Russian twist', { note: 'Gerakan rotasi sulit diukur dari sudut 2D' }),
  shoulder_press: entry('Shoulder press'),
  squat: entry('Squat'),
  t_bar_row: entry('T-bar row'),
  tricep_dips: entry('Tricep dips'),
  tricep_pushdown: entry('Tricep pushdown'),
});

/** Display name for a label, falling back to the label itself for anything not in the catalog. */
export const exerciseName = (label) => CATALOG[label]?.name ?? label;
