import { test } from 'node:test';
import assert from 'node:assert/strict';
import { calculateAngle, angleFromVertical } from '../../../src/core/geometry/angles.js';

const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 0.01, `${actual} ≉ ${expected}`);

test('returns 90 for a right angle', () => {
  close(calculateAngle({ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 }), 90);
});

test('returns 180 for a straight line', () => {
  close(calculateAngle({ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 0, y: 2 }), 180);
});

test('returns null when any point is missing', () => {
  assert.equal(calculateAngle(null, { x: 0, y: 1 }, { x: 1, y: 1 }), null);
});

test('returns null when two points coincide', () => {
  assert.equal(calculateAngle({ x: 1, y: 1 }, { x: 1, y: 1 }, { x: 2, y: 2 }), null);
});

test('angleFromVertical is 0 for an upright segment', () => {
  close(angleFromVertical({ x: 5, y: 0 }, { x: 5, y: 10 }), 0);
});

test('angleFromVertical is 45 for a diagonal segment', () => {
  close(angleFromVertical({ x: 0, y: 0 }, { x: 10, y: 10 }), 45);
});
