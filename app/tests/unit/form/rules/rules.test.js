import { test } from 'node:test';
import assert from 'node:assert/strict';

import { RULES, rulesFor } from '../../../../src/core/form/rules/index.js';
import {
  WARNING, measureExercise, evaluateForm, brokenRules, inPosition, createRepCheck, updateRepCheck,
  assessPoseQuality, QUALITY,
} from '../../../../src/core/form/measure.js';
import { EXERCISES } from '../../../../src/core/exercises.js';

const SIZE = { width: 1000, height: 1000 };
const INDEX = {
  ear: [7, 8], shoulder: [11, 12], elbow: [13, 14], wrist: [15, 16], hip: [23, 24], knee: [25, 26], ankle: [27, 28],
};
const deg = (d) => (d * Math.PI) / 180;

/** A 33-landmark pose from named points; both sides share them unless `right` overrides. */
function pose(points, { visibility = { left: 0.9, right: 0.9 }, right = {} } = {}) {
  const landmarks = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.1 }));
  for (const [name, [li, ri]] of Object.entries(INDEX)) {
    const left = points[name] ?? [0.5, 0.5];
    const r = right[name] ?? left;
    landmarks[li] = { x: left[0], y: left[1], z: 0, visibility: visibility.left };
    landmarks[ri] = { x: r[0] + 0.01, y: r[1], z: 0, visibility: visibility.right };
  }
  return landmarks;
}

const broken = (label, p) => brokenRules(evaluateForm(p, RULES[label], SIZE));

// ---------------------------------------------------------------- the registry

test('rules exist only for the exercises with a clear 2D check', () => {
  assert.deepEqual(Object.keys(RULES).sort(),
    ['barbell_biceps_curl', 'deadlift', 'lateral_raise', 'plank', 'push_up', 'shoulder_press', 'squat']);
  assert.equal(rulesFor('lat_pulldown'), null);
});

test('every rule is declarative: points, kind, limit, message', () => {
  const kinds = new Set(['angle', 'angleFromVertical', 'heightAbove', 'lineOffset']);
  for (const [label, set] of Object.entries(RULES)) {
    assert.ok(['side', 'any'].includes(set.view), label);
    for (const rule of set.rules) {
      assert.ok(kinds.has(rule.measure.kind), `${rule.id} kind`);
      assert.ok(rule.measure.points.every((p) => p in INDEX), `${rule.id} points`);
      assert.ok('min' in rule.limit || 'max' in rule.limit, `${rule.id} limit`);
      assert.ok(rule.message.length > 0, `${rule.id} message`);
    }
  }
});

// ---------------------------------------------------------------- the original two rules, kept

const squatPose = (lean) => pose({
  hip: [0.5, 0.6], shoulder: [0.5 + 0.3 * Math.sin(deg(lean)), 0.6 - 0.3 * Math.cos(deg(lean))],
  knee: [0.55, 0.75], ankle: [0.5, 0.9], elbow: [0.5, 0.4], wrist: [0.5, 0.5], ear: [0.5, 0.2],
});

const pushUpPose = (sag) => pose({
  shoulder: [0.3, 0.5], hip: [0.55, 0.5 + sag], ankle: [0.8, 0.5],
  elbow: [0.3, 0.6], wrist: [0.3, 0.7], knee: [0.68, 0.5], ear: [0.25, 0.48],
});

test('squat and push-up warn exactly when the original measureExercise did', () => {
  for (const lean of [10, 30, 44, 46, 60, 80]) {
    const legacy = measureExercise(squatPose(lean), EXERCISES.squat, SIZE).warnings.includes(WARNING.TORSO_LEAN);
    assert.equal(broken('squat', squatPose(lean)).length > 0, legacy, `squat lean ${lean}°`);
  }
  for (const sag of [-0.05, 0, 0.02, 0.04, 0.06, 0.1]) {
    const legacy = measureExercise(pushUpPose(sag), EXERCISES.pushup, SIZE).warnings.includes(WARNING.BODY_LINE);
    assert.equal(broken('push_up', pushUpPose(sag)).length > 0, legacy, `push-up sag ${sag}`);
  }
  assert.equal(RULES.squat.rules[0].message, WARNING.TORSO_LEAN);
  assert.equal(RULES.push_up.rules[0].message, WARNING.BODY_LINE);
});

test('squat lean crosses its limit at 45°', () => {
  assert.deepEqual(broken('squat', squatPose(40)), []);
  assert.deepEqual(broken('squat', squatPose(50)), ['squat-torso-lean']);
});

// ---------------------------------------------------------------- the new rules

const curlPose = (elbowForward) => pose({
  shoulder: [0.5, 0.3], hip: [0.5, 0.6],
  elbow: [0.5 + 0.15 * Math.sin(deg(elbowForward)), 0.3 + 0.15 * Math.cos(deg(elbowForward))],
  wrist: [0.6, 0.35], knee: [0.5, 0.75], ankle: [0.5, 0.9], ear: [0.5, 0.2],
});

test('biceps curl warns when the elbow swings forward past 30°', () => {
  assert.deepEqual(broken('barbell_biceps_curl', curlPose(15)), []);
  assert.deepEqual(broken('barbell_biceps_curl', curlPose(45)), ['curl-elbow-forward']);
});

const raisePose = (wristAboveShoulder) => pose({
  shoulder: [0.5, 0.3], hip: [0.5, 0.6], elbow: [0.65, 0.3], wrist: [0.8, 0.3 - 0.3 * wristAboveShoulder],
  knee: [0.5, 0.75], ankle: [0.5, 0.9], ear: [0.5, 0.2],
});

test('lateral raise warns when the hands rise clearly above the shoulders', () => {
  assert.deepEqual(broken('lateral_raise', raisePose(0)), [], 'shoulder height is the target');
  assert.deepEqual(broken('lateral_raise', raisePose(0.1)), []);
  assert.deepEqual(broken('lateral_raise', raisePose(0.4)), ['lateral-raise-too-high']);
});

const deadliftPose = (earDrop) => pose({
  hip: [0.6, 0.5], shoulder: [0.4, 0.4], ear: [0.3, 0.35 + earDrop],
  knee: [0.6, 0.7], ankle: [0.6, 0.9], elbow: [0.4, 0.5], wrist: [0.4, 0.6],
});

test('deadlift warns when the shoulder leaves the ear–hip line', () => {
  assert.deepEqual(broken('deadlift', deadliftPose(0)), [], 'ear, shoulder, hip in one line');
  assert.deepEqual(broken('deadlift', deadliftPose(0.12)), ['deadlift-rounded-back']);
});

function pressPose(elbowAngle) {
  const elbow = [0.5, 0.3];
  return pose({
    elbow, shoulder: [0.5, 0.42],
    wrist: [elbow[0] + 0.12 * Math.sin(deg(elbowAngle)), elbow[1] + 0.12 * Math.cos(deg(elbowAngle))],
    hip: [0.5, 0.7], knee: [0.5, 0.82], ankle: [0.5, 0.95], ear: [0.5, 0.35],
  });
}

function pressRep(peakAngle, check = createRepCheck()) {
  let state = check;
  for (const angle of [90, 120, peakAngle, 120]) {
    ({ state } = updateRepCheck(state, evaluateForm(pressPose(angle), RULES.shoulder_press, SIZE), RULES.shoulder_press, false));
  }
  return updateRepCheck(state, evaluateForm(pressPose(90), RULES.shoulder_press, SIZE), RULES.shoulder_press, true);
}

test('shoulder press is judged once per rep, on its straightest moment', () => {
  assert.deepEqual(broken('shoulder_press', pressPose(100)), [], 'never "broken" frame by frame');
  assert.deepEqual(pressRep(175).failed, []);
  assert.deepEqual(pressRep(140).failed.map((f) => f.id), ['shoulder-press-lockout']);
});

test('the rep check starts clean after each rep', () => {
  const { state } = pressRep(140);
  assert.deepEqual(pressRep(175, state).failed, []);
  assert.deepEqual(updateRepCheck(createRepCheck(), { results: [] }, RULES.squat, true).failed, []);
});

const plankPose = (offset) => {
  const shoulder = [0.3, 0.5];
  const ankle = [0.8, 0.55];
  const length = Math.hypot(0.5, 0.05);
  return pose({
    shoulder, ankle, hip: [0.55, 0.525 + offset * length],
    knee: [0.68, 0.54], elbow: [0.3, 0.6], wrist: [0.32, 0.7], ear: [0.25, 0.48],
  });
};

test('plank: a straight body is in position, sagging or piking hips are not', () => {
  const straight = evaluateForm(plankPose(0), RULES.plank, SIZE);
  assert.equal(inPosition(straight), true);
  assert.deepEqual(broken('plank', plankPose(0.15)), ['plank-hips-low']);
  assert.deepEqual(broken('plank', plankPose(-0.2)), ['plank-hips-high']);
  assert.equal(inPosition(evaluateForm(plankPose(0.15), RULES.plank, SIZE)), false);
});

// ---------------------------------------------------------------- sides and edge cases

test('the better-seen side is measured, and nothing when neither side is seen', () => {
  const leftHidden = squatPose(10).map((p, i) => ([11, 23, 25, 27].includes(i) ? { ...p, visibility: 0.1 } : p));
  assert.equal(evaluateForm(leftHidden, RULES.squat, SIZE).side, 'right');
  const hidden = squatPose(10).map((p) => ({ ...p, visibility: 0.2 }));
  assert.deepEqual(evaluateForm(hidden, RULES.squat, SIZE), { measurable: false, side: null, results: [] });
  assert.equal(inPosition(evaluateForm(null, RULES.plank, SIZE)), false);
  assert.equal(evaluateForm(squatPose(10), null, SIZE).measurable, false);
});

test('a rule stays silent about points it cannot see', () => {
  // Hips and ankles guessed from outside the frame: the arm points pick the side, but the body-line
  // rule needs the hip and ankle, so it must not warn — the original app did.
  const outOfFrame = pushUpPose(0.1).map((p, i) => ([23, 24, 27, 28].includes(i) ? { ...p, visibility: 0.2 } : p));
  assert.deepEqual(broken('push_up', outOfFrame), []);
  assert.equal(evaluateForm(outOfFrame, RULES.push_up, SIZE).results[0].value, null);
});

test('the plank timer does not run on a hip it cannot see', () => {
  const hipHidden = plankPose(0).map((p, i) => ([23, 24].includes(i) ? { ...p, visibility: 0.2 } : p));
  assert.equal(inPosition(evaluateForm(hipHidden, RULES.plank, SIZE)), false);
});

test('a degenerate pose measures nothing instead of warning', () => {
  const collapsed = pose({ shoulder: [0.5, 0.5], hip: [0.5, 0.5], elbow: [0.5, 0.5], wrist: [0.5, 0.5],
    ankle: [0.5, 0.5], knee: [0.5, 0.5], ear: [0.5, 0.5] });
  for (const label of Object.keys(RULES)) assert.deepEqual(broken(label, collapsed), [], label);
});

// ---------------------------------------------------------------- pose quality

test('no pose, a body half out of frame, and a frontal camera each get their hint', () => {
  assert.equal(assessPoseQuality(null, SIZE), QUALITY.NO_POSE);
  const hidden = squatPose(10).map((p) => ({ ...p, visibility: 0.2 }));
  assert.equal(assessPoseQuality(hidden, SIZE), QUALITY.PARTIAL);
  const frontal = pose({ shoulder: [0.35, 0.3], hip: [0.42, 0.6], knee: [0.42, 0.75], ankle: [0.42, 0.9] },
    { right: { shoulder: [0.64, 0.3], hip: [0.57, 0.6], knee: [0.57, 0.75], ankle: [0.57, 0.9] } });
  assert.equal(assessPoseQuality(frontal, SIZE, { view: 'side' }), QUALITY.FRONTAL);
  assert.equal(assessPoseQuality(frontal, SIZE, { view: 'any' }), null, 'lateral raise is fine from the front');
  assert.equal(assessPoseQuality(squatPose(10), SIZE, { view: 'side' }), null, 'a side view is what we want');
});

test('the visibility check follows the points the exercise needs', () => {
  const legsHidden = squatPose(10).map((p, i) => ([25, 26, 27, 28].includes(i) ? { ...p, visibility: 0.1 } : p));
  assert.equal(assessPoseQuality(legsHidden, SIZE), null, 'seated upper-body work needs no legs');
  assert.equal(assessPoseQuality(legsHidden, SIZE, { points: RULES.squat.sidePoints }), QUALITY.PARTIAL);
});
