/**
 * Deterministic synthetic MediaPipe pose streams with known rep times, for counter tests.
 * The squat mirrors `_squat_pose` in ml/src/repcount/features/golden.py; the curl moves only the
 * arms, so a counter that secretly depends on the legs fails it.
 */

/** mulberry32: a tiny seeded PRNG so every test run sees the same noise. */
export function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal samples (Box–Muller) from a uniform generator. */
export function gaussian(uniform) {
  return () => Math.sqrt(-2 * Math.log(uniform() || 1e-12)) * Math.cos(2 * Math.PI * uniform());
}

export function squatPose(depth) {
  const hipY = 0.55 + 0.12 * depth;
  const shoulderY = 0.25 + 0.16 * depth;
  const lean = 0.04 * depth;
  return {
    0: [0.5 + lean, 0.1 + 0.18 * depth],
    11: [0.45 + lean, shoulderY], 12: [0.55 + lean, shoulderY],
    13: [0.43 + lean, shoulderY + 0.14], 14: [0.57 + lean, shoulderY + 0.14],
    15: [0.42 + lean, shoulderY + 0.28], 16: [0.58 + lean, shoulderY + 0.28],
    23: [0.46, hipY], 24: [0.54, hipY],
    25: [0.45 + 0.03 * depth, 0.76], 26: [0.55 + 0.03 * depth, 0.76],
    27: [0.45, 0.95], 28: [0.55, 0.95],
  };
}

export function curlPose(depth) {
  // Upper arms stay put; forearms swing from hanging (depth 0) up towards the shoulders (depth 1).
  const swing = depth * 0.8 * Math.PI;
  const forearm = (elbowX, side) => [elbowX + side * 0.12 * Math.sin(swing), 0.4 + 0.12 * Math.cos(swing)];
  return {
    0: [0.5, 0.1],
    11: [0.45, 0.25], 12: [0.55, 0.25],
    13: [0.44, 0.4], 14: [0.56, 0.4],
    15: forearm(0.44, -0.3), 16: forearm(0.56, 0.3),
    23: [0.46, 0.55], 24: [0.54, 0.55],
    25: [0.46, 0.75], 26: [0.54, 0.75],
    27: [0.46, 0.95], 28: [0.54, 0.95],
  };
}

function toLandmarks(points, jitter, noise) {
  return Array.from({ length: 33 }, (_, i) => {
    const point = points[i];
    if (!point) return { x: 0.5, y: 0.5, z: 0, visibility: 0.05 };
    return { x: point[0] + jitter * noise(), y: point[1] + jitter * noise(), z: 0, visibility: 0.9 };
  });
}

/**
 * A stream of `segments`, each `{ reps, repSeconds }`, framed by rests at depth 0.
 * Rep k of a segment completes when depth returns to 0 — the moment labels/README.md marks.
 * @returns {{ frames: object[][], timestampsMs: number[], repTimesMs: number[], size: object }}
 */
export function poseStream({
  segments, pose = squatPose, fps = 30, restSeconds = 1, tailSeconds = 1, jitter = 0.0015, seed = 7,
}) {
  const noise = gaussian(random(seed));
  const depths = [];
  const repTimesMs = [];
  const pushRest = (seconds) => { for (let i = 0; i < Math.round(seconds * fps); i += 1) depths.push(0); };

  pushRest(restSeconds);
  for (const { reps, repSeconds } of segments) {
    const perRep = Math.round(repSeconds * fps);
    for (let r = 0; r < reps; r += 1) {
      for (let i = 1; i <= perRep; i += 1) depths.push((1 - Math.cos((2 * Math.PI * i) / perRep)) / 2);
      repTimesMs.push(Math.round(((depths.length - 1) * 1000) / fps));
    }
  }
  pushRest(tailSeconds);

  return {
    frames: depths.map((depth) => toLandmarks(pose(depth), jitter, noise)),
    timestampsMs: depths.map((_, i) => Math.round((i * 1000) / fps)),
    repTimesMs,
    size: { width: 1280, height: 720 },
  };
}
