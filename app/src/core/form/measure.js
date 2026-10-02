import { calculateAngle, angleFromVertical } from '../geometry/angles.js';
import { LANDMARK } from '../exercises.js';

const MIN_VISIBILITY = 0.5;

export const WARNING = Object.freeze({
  TORSO_LEAN: 'Punggung terlalu membungkuk',
  BODY_LINE: 'Jaga badan tetap lurus',
});

// MediaPipe landmarks are normalized to [0, 1]; convert to pixels so angles
// are not distorted by the frame's aspect ratio.
function toPixel(landmark, { width, height }) {
  return { x: landmark.x * width, y: landmark.y * height };
}

function minVisibility(pose, indices) {
  return Math.min(...indices.map((i) => pose[i]?.visibility ?? 0));
}

// The camera usually sees one side better than the other; use that side.
function pickVisibleSide(pose, sides) {
  const [best] = Object.values(sides)
    .map((side) => ({ side, visibility: minVisibility(pose, side.joint) }))
    .sort((a, b) => b.visibility - a.visibility);
  return best.visibility >= MIN_VISIBILITY ? best.side : null;
}

function checkForm(points, side, exercise) {
  const warnings = [];

  if (side.torso && exercise.maxTorsoLean) {
    const [shoulder, hip] = points(side.torso);
    if (angleFromVertical(shoulder, hip) > exercise.maxTorsoLean) warnings.push(WARNING.TORSO_LEAN);
  }

  if (side.bodyLine && exercise.minBodyLine) {
    const bodyAngle = calculateAngle(...points(side.bodyLine));
    if (bodyAngle !== null && bodyAngle < exercise.minBodyLine) warnings.push(WARNING.BODY_LINE);
  }

  return warnings;
}

/**
 * Measures one pose for an exercise.
 * @returns {{ angle: number|null, warnings: string[] }}
 */
export function measureExercise(pose, exercise, frameSize) {
  if (!pose) return { angle: null, warnings: [] };

  const side = pickVisibleSide(pose, exercise.sides);
  if (!side) return { angle: null, warnings: [] };

  const points = (indices) => indices.map((i) => toPixel(pose[i], frameSize));
  return {
    angle: calculateAngle(...points(side.joint)),
    warnings: checkForm(points, side, exercise),
  };
}

// ---------------------------------------------------------------- declarative rules (core/form/rules/)

const L = LANDMARK;
const POINTS = Object.freeze({
  ear: [L.LEFT_EAR, L.RIGHT_EAR],
  shoulder: [L.LEFT_SHOULDER, L.RIGHT_SHOULDER],
  elbow: [L.LEFT_ELBOW, L.RIGHT_ELBOW],
  wrist: [L.LEFT_WRIST, L.RIGHT_WRIST],
  hip: [L.LEFT_HIP, L.RIGHT_HIP],
  knee: [L.LEFT_KNEE, L.RIGHT_KNEE],
  ankle: [L.LEFT_ANKLE, L.RIGHT_ANKLE],
});
const SIDES = Object.freeze(['left', 'right']);
const sideIndex = (side) => (side === 'left' ? 0 : 1);

/** The better-seen side over `names`, as the original measureExercise chose it (ties → left). */
function pickSide(pose, names) {
  const visibility = (side) => minVisibility(pose, names.map((name) => POINTS[name][sideIndex(side)]));
  const [best] = SIDES.map((side) => ({ side, v: visibility(side) })).sort((a, b) => b.v - a.v);
  return best.v >= MIN_VISIBILITY ? best.side : null;
}

/** Height of the hip below the shoulder–ankle line, in line lengths (positive = below). */
function lineOffset(a, p, b) {
  if (Math.abs(b.x - a.x) < 1e-9) return null;
  const lineY = a.y + ((b.y - a.y) * (p.x - a.x)) / (b.x - a.x);
  return (p.y - lineY) / Math.hypot(b.x - a.x, b.y - a.y);
}

const MEASURES = Object.freeze({
  angle: ([a, b, c]) => calculateAngle(a, b, c),
  angleFromVertical: ([top, bottom]) => angleFromVertical(top, bottom),
  heightAbove: ([p, ref], torso) => (torso > 0 ? (ref.y - p.y) / torso : null),
  lineOffset: ([a, p, b]) => lineOffset(a, p, b),
});

const breaks = ({ min, max }, value) => value !== null && Number.isFinite(value)
  && ((max !== undefined && value > max) || (min !== undefined && value < min));

/**
 * Measure every rule of a rule set on one pose (pixel space, the better-seen side).
 * @returns {{ measurable: boolean, side: string|null, results: Array<{ id: string, message: string,
 *   value: number|null, broken: boolean, per: string }> }} per-rep rules are measured but never
 *   "broken" here — updateRepCheck judges them when a rep ends
 */
export function evaluateForm(pose, ruleSet, frameSize) {
  if (!pose || !ruleSet) return { measurable: false, side: null, results: [] };
  const side = pickSide(pose, ruleSet.sidePoints);
  if (!side) return { measurable: false, side: null, results: [] };

  const at = (name) => toPixel(pose[POINTS[name][sideIndex(side)]], frameSize);
  const torso = Math.hypot(at('shoulder').x - at('hip').x, at('shoulder').y - at('hip').y);
  const results = ruleSet.rules.map((rule) => {
    const value = MEASURES[rule.measure.kind](rule.measure.points.map(at), torso);
    const per = rule.per ?? 'frame';
    return { id: rule.id, message: rule.message, value, per, broken: per === 'frame' && breaks(rule.limit, value) };
  });
  return { measurable: true, side, results };
}

/** Ids of the per-frame rules broken in this evaluation (input for core/form/debounce.js). */
export const brokenRules = (evaluation) => evaluation.results.filter((r) => r.broken).map((r) => r.id);

/** Whether the pose satisfies every rule: the plank timer runs only while this holds. */
export const inPosition = (evaluation) => evaluation.measurable && evaluation.results.every((r) => !r.broken);

export function createRepCheck() {
  return Object.freeze({ extremes: Object.freeze({}) });
}

/**
 * Track per-rep rules over a rep and judge them when it ends. A `min` limit is judged on the largest
 * value seen during the rep (e.g. lockout: the straightest the elbow got), a `max` limit on the smallest.
 * @returns {{ state: object, failed: Array<{ id: string, message: string }> }}
 */
export function updateRepCheck(state, evaluation, ruleSet, repCompleted) {
  const perRep = (ruleSet?.rules ?? []).filter((rule) => rule.per === 'rep');
  const extremes = { ...state.extremes };
  for (const rule of perRep) {
    const value = evaluation.results.find((r) => r.id === rule.id)?.value;
    if (value === null || value === undefined) continue;
    const keep = rule.limit.min !== undefined ? Math.max : Math.min;
    extremes[rule.id] = extremes[rule.id] === undefined ? value : keep(extremes[rule.id], value);
  }
  if (!repCompleted) return { state: Object.freeze({ extremes: Object.freeze(extremes) }), failed: [] };
  const failed = perRep.filter((rule) => extremes[rule.id] !== undefined && breaks(rule.limit, extremes[rule.id]))
    .map((rule) => ({ id: rule.id, message: rule.message }));
  return { state: createRepCheck(), failed };
}

// ---------------------------------------------------------------- pose quality hints

export const QUALITY = Object.freeze({
  NO_POSE: 'Tubuh tidak terdeteksi — mundur sedikit dari kamera',
  PARTIAL: 'Tubuh tidak terlihat penuh',
  FRONTAL: 'Posisikan kamera dari samping',
});
// In a side view the shoulders nearly overlap; facing the camera they are about a torso length apart.
const FRONTAL_SHOULDER_RATIO = 0.55;

/**
 * One hint about the camera set-up, or null when the pose is good enough.
 * @param {{ points?: string[], view?: 'side'|'any' }} needs the points the current exercise uses and
 *   whether it is judged from the side; by default shoulders and hips, which every feature needs
 */
export function assessPoseQuality(pose, frameSize, { points = ['shoulder', 'hip'], view = 'any' } = {}) {
  if (!pose) return QUALITY.NO_POSE;
  const side = pickSide(pose, points);
  if (!side) return QUALITY.PARTIAL;
  if (view !== 'side') return null;
  const px = (index) => toPixel(pose[index], frameSize);
  const shoulderWidth = Math.abs(px(L.LEFT_SHOULDER).x - px(L.RIGHT_SHOULDER).x);
  const s = px(POINTS.shoulder[sideIndex(side)]);
  const h = px(POINTS.hip[sideIndex(side)]);
  const torso = Math.hypot(s.x - h.x, s.y - h.y);
  return torso > 0 && shoulderWidth / torso > FRONTAL_SHOULDER_RATIO ? QUALITY.FRONTAL : null;
}
