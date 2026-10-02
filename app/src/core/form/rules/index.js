import squat from './squat.js';
import pushUp from './pushUp.js';
import barbellBicepsCurl from './barbellBicepsCurl.js';
import lateralRaise from './lateralRaise.js';
import deadlift from './deadlift.js';
import shoulderPress from './shoulderPress.js';
import plank from './plank.js';

/**
 * Form rules by dataset label. Only exercises whose rules are clear and measurable from one 2D view
 * are here (docs/FORM_RULES.md); every other exercise is counted without form feedback.
 */
export const RULES = Object.freeze({
  squat,
  push_up: pushUp,
  barbell_biceps_curl: barbellBicepsCurl,
  lateral_raise: lateralRaise,
  deadlift,
  shoulder_press: shoulderPress,
  plank,
});

export const rulesFor = (label) => RULES[label] ?? null;
