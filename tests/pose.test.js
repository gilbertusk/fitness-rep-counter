import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureExercise } from '../src/core/pose.js';
import { EXERCISES } from '../src/core/exercises.js';

const FRAME = { width: 100, height: 100 };

// Builds a 33-point MediaPipe pose where only the listed indices are set.
function makePose(points, visibility = 0.9) {
  const pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
  for (const [index, [x, y]] of Object.entries(points)) {
    pose[index] = { x, y, visibility };
  }
  return pose;
}

// Standing: shoulder, hip, knee, ankle stacked vertically on both sides.
const standing = makePose({
  11: [0.4, 0.2], 23: [0.4, 0.5], 25: [0.4, 0.7], 27: [0.4, 0.9],
  12: [0.6, 0.2], 24: [0.6, 0.5], 26: [0.6, 0.7], 28: [0.6, 0.9],
});

test('measures a straight knee as ~180 degrees when standing', () => {
  const { angle } = measureExercise(standing, EXERCISES.squat, FRAME);
  assert.ok(angle > 175);
});

test('reports no form warnings for an upright squat', () => {
  assert.deepEqual(measureExercise(standing, EXERCISES.squat, FRAME).warnings, []);
});

test('warns when the torso leans too far forward in a squat', () => {
  const leaning = makePose({
    11: [0.1, 0.45], 23: [0.4, 0.5], 25: [0.5, 0.7], 27: [0.4, 0.9],
  });
  const { warnings } = measureExercise(leaning, EXERCISES.squat, FRAME);
  assert.ok(warnings.includes('Punggung terlalu membungkuk'));
});

test('warns when the hips sag during a push-up', () => {
  const sagging = makePose({
    11: [0.2, 0.5], 13: [0.2, 0.6], 15: [0.2, 0.7], 23: [0.5, 0.65], 27: [0.8, 0.5],
  });
  const { warnings } = measureExercise(sagging, EXERCISES.pushup, FRAME);
  assert.ok(warnings.includes('Jaga badan tetap lurus'));
});

test('returns a null angle when the relevant joints are not visible', () => {
  const hidden = makePose({ 23: [0.4, 0.5], 25: [0.4, 0.7], 27: [0.4, 0.9] }, 0.1);
  assert.equal(measureExercise(hidden, EXERCISES.squat, FRAME).angle, null);
});

test('returns a null angle when no pose is detected', () => {
  assert.equal(measureExercise(undefined, EXERCISES.squat, FRAME).angle, null);
});
