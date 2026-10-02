import { test } from 'node:test';
import assert from 'node:assert/strict';

import { COUNTERS, toLandmarks, runCounter } from '../lib/counters.js';
import { poseStream } from '../../../app/tests/fixtures/syntheticPose.js';

const SIZE = { width: 1280, height: 720 };

/** A deep squat built in pixel space: the knee closes from 180° to 85°, past the threshold counter's 95°. */
function deepSquatPose(depth) {
  const theta = (depth * 95 * Math.PI) / 180;
  const knee = [640, 504];
  const hip = [knee[0] + 150 * Math.sin(theta), knee[1] - 150 * Math.cos(theta)];
  const shoulder = [hip[0], hip[1] - 200];
  const norm = ([px, py]) => [px / SIZE.width, py / SIZE.height];
  const side = (dx) => ({
    shoulder: norm([shoulder[0] + dx, shoulder[1]]), elbow: norm([shoulder[0] + dx, shoulder[1] + 90]),
    wrist: norm([shoulder[0] + dx, shoulder[1] + 170]), hip: norm([hip[0] + dx, hip[1]]),
    knee: norm([knee[0] + dx, knee[1]]), ankle: norm([knee[0] + dx, knee[1] + 158]),
  });
  const left = side(-20);
  const right = side(20);
  return {
    0: norm([shoulder[0], shoulder[1] - 60]),
    11: left.shoulder, 12: right.shoulder, 13: left.elbow, 14: right.elbow, 15: left.wrist, 16: right.wrist,
    23: left.hip, 24: right.hip, 25: left.knee, 26: right.knee, 27: left.ankle, 28: right.ankle,
  };
}

/** A synthetic stream in the data/keypoints_json layout the harness reads. */
function asJsonVideo(stream, label) {
  return {
    video_id: `${label}_1`, label, width: stream.size.width, height: stream.size.height,
    timestamps_ms: stream.timestampsMs,
    landmarks: stream.frames.map((frame) => frame.map((p) => [p.x, p.y, p.z, p.visibility])),
  };
}

test('the registry offers the generic signals, the naive floor and the threshold baseline', () => {
  assert.deepEqual(Object.keys(COUNTERS).sort(), ['generic', 'generic-angle', 'naive-peaks', 'threshold']);
  Object.values(COUNTERS).forEach((counter) => {
    ['supports', 'create', 'update', 'result'].forEach((fn) => assert.equal(typeof counter[fn], 'function'));
  });
});

test('a JSON frame becomes MediaPipe-shaped landmarks, and a missing pose stays missing', () => {
  assert.deepEqual(toLandmarks([[0.1, 0.2, 0.3, 0.9]]), [{ x: 0.1, y: 0.2, z: 0.3, visibility: 0.9 }]);
  assert.equal(toLandmarks(null), null);
});

test('generic and naive counters stream a whole video and report rep times', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], jitter: 0.0005 });
  for (const name of ['generic', 'generic-angle', 'naive-peaks']) {
    const result = runCounter(name, asJsonVideo(stream, 'squat'));
    assert.equal(result.count, 3, name);
    assert.equal(result.repTimes.length, 3, name);
    assert.ok(result.msPerFrame >= 0);
  }
});

test('the threshold counter counts squats deep enough for its hand-set angles', () => {
  const stream = poseStream({ segments: [{ reps: 4, repSeconds: 2 }], pose: deepSquatPose, jitter: 0.0005 });
  const result = runCounter('threshold', asJsonVideo(stream, 'squat'));
  assert.equal(result.count, 4);
  result.repTimes.forEach((t, i) => assert.ok(Math.abs(t - stream.repTimesMs[i]) < 800, `rep ${i} at ${t}`));
});

test('the threshold counter is N/A outside squat and push-up, never a silent zero', () => {
  const stream = poseStream({ segments: [{ reps: 2, repSeconds: 2 }] });
  assert.equal(runCounter('threshold', asJsonVideo(stream, 'lat_pulldown')), null);
  assert.equal(COUNTERS.threshold.supports('push_up'), true);
});

test('tuning parameters reach the counter', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], jitter: 0.0005 });
  assert.equal(runCounter('generic', asJsonVideo(stream, 'squat'), { minAmplitude: 1000 }).count, 0);
  assert.equal(runCounter('naive-peaks', asJsonVideo(stream, 'squat'), { prominenceDegrees: 179 }).count, 0);
});

test('an unknown counter name fails loudly with the known names', () => {
  assert.throws(() => runCounter('magic', { label: 'squat', landmarks: [] }), /unknown counter magic; known: generic/);
});

test('an empty video gives zero reps and no division by zero', () => {
  const result = runCounter('generic', { label: 'squat', width: 640, height: 480, timestamps_ms: [], landmarks: [] });
  assert.deepEqual(result, { count: 0, repTimes: [], msPerFrame: 0 });
});
