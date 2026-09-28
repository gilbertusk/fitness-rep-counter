import { calculateAngle, angleFromVertical } from '../geometry/angles.js';

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
