import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULTS, SIGNAL, estimatePeriod, createSignalCounter, updateSignalCounter, resetZones, stepWindow,
  principalComponent, busiestAngle, createGenericCounter, updateGenericCounter,
} from '../../../src/core/counting/genericCounter.js';
import { random, gaussian, poseStream, curlPose } from '../../fixtures/syntheticPose.js';

const FPS = DEFAULTS.sampleFps;

/**
 * A resting-then-moving signal like a real set: flat at 0, then `reps` cycles of
 * amplitude·(1 − cos)/2 (so each rep returns to the start), then flat again.
 * Returns the samples and the times at which each cycle completes.
 */
function repSignal({ reps, repSeconds, amplitude = 0.5, restSeconds = 1, noise = 0, seed = 3, offset = 0 }) {
  const rand = gaussian(random(seed));
  const values = [];
  const completions = [];
  const rest = () => { for (let i = 0; i < restSeconds * FPS; i += 1) values.push(offset + noise * rand()); };
  rest();
  const perRep = Math.round(repSeconds * FPS);
  for (let r = 0; r < reps; r += 1) {
    for (let i = 1; i <= perRep; i += 1) {
      values.push(offset + (amplitude * (1 - Math.cos((2 * Math.PI * i) / perRep))) / 2 + noise * rand());
    }
    completions.push(((values.length - 1) * 1000) / FPS);
  }
  rest();
  return { values, completions };
}

const countSignal = (values, options = { minAmplitude: 0.05 }) => values.reduce(
  (state, value, i) => updateSignalCounter(state, value, (i * 1000) / FPS), createSignalCounter(options));

const countStream = (stream, options = {}) => stream.frames.reduce(
  (state, frame, i) => updateGenericCounter(state, frame, stream.timestampsMs[i], stream.size), createGenericCounter(options));

// ---------------------------------------------------------------- signal level

test('N clean cycles count as N reps across frequencies and amplitudes', () => {
  for (const repSeconds of [0.8, 1.5, 3]) {
    for (const amplitude of [0.1, 0.5, 2]) {
      for (const reps of [1, 3, 8]) {
        const { values } = repSignal({ reps, repSeconds, amplitude });
        assert.equal(countSignal(values).count, reps, `${reps} reps of ${repSeconds}s at amplitude ${amplitude}`);
      }
    }
  }
});

test('the very first rep counts: no period lock is needed', () => {
  const { values } = repSignal({ reps: 1, repSeconds: 2 });
  assert.equal(countSignal(values).count, 1);
});

test('a rep is reported close to the moment it returns to the start', () => {
  const { values, completions } = repSignal({ reps: 5, repSeconds: 1.5 });
  const { repTimes } = countSignal(values);
  repTimes.forEach((t, i) => assert.ok(Math.abs(t - completions[i]) <= 400,
    `rep ${i}: reported ${t} ms, completed ${completions[i]} ms`));
});

test('noise on top of the cycles never double counts', () => {
  // Noise of σ 0.05 spans ±0.08 while resting, so the stillness floor has to sit above that —
  // which is exactly the job minAmplitude does on real MediaPipe jitter.
  for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
    const { values } = repSignal({ reps: 6, repSeconds: 1.2, amplitude: 0.5, noise: 0.05, seed });
    assert.equal(countSignal(values, { minAmplitude: 0.2 }).count, 6, `seed ${seed}`);
  }
});

test('noise below the stillness floor during rests adds nothing between sets', () => {
  const rand = gaussian(random(11));
  const rest = Array.from({ length: 3 * FPS }, () => 0.03 * rand());
  const { values } = repSignal({ reps: 3, repSeconds: 1.5, noise: 0.03, seed: 11 });
  assert.equal(countSignal([...rest, ...values, ...rest], { minAmplitude: 0.15 }).count, 3);
});

test('a flat signal and noise below the stillness threshold count nothing', () => {
  assert.equal(countSignal(new Array(200).fill(0.7)).count, 0);
  const rand = gaussian(random(9));
  assert.equal(countSignal(Array.from({ length: 300 }, () => 0.7 + 0.008 * rand())).count, 0);
});

test('a change of tempo mid-set is still counted rep by rep', () => {
  const slow = repSignal({ reps: 4, repSeconds: 2, restSeconds: 1 }).values;
  const fast = repSignal({ reps: 4, repSeconds: 0.8, restSeconds: 0 }).values;
  assert.equal(countSignal([...slow.slice(0, -FPS), ...fast]).count, 8);
});

test('the counter is indifferent to the signal offset and to which way a rep goes', () => {
  const { values } = repSignal({ reps: 4, repSeconds: 1.2, offset: 5 });
  assert.equal(countSignal(values).count, 4);
  assert.equal(countSignal(values.map((v) => -v)).count, 4);
});

test('the period is found once a few cycles are in the window', () => {
  const { values } = repSignal({ reps: 6, repSeconds: 1 });
  const state = countSignal(values.slice(0, FPS * 6));
  assert.ok(Math.abs(state.periodSeconds - 1) < 0.15, `period ${state.periodSeconds}`);
});

test('the period estimate ignores a flat signal and prefers the fundamental over its double', () => {
  assert.equal(estimatePeriod(new Array(60).fill(1), 8, 0.4), null);
  const wave = Array.from({ length: 60 }, (_, i) => Math.cos((2 * Math.PI * i) / 12));
  assert.equal(estimatePeriod(wave, 8, 0.4), 12);
});

test('non-finite samples are skipped without touching the state', () => {
  const state = createSignalCounter();
  assert.equal(updateSignalCounter(state, NaN, 0), state);
  assert.equal(updateSignalCounter(state, Infinity, 0), state);
});

test('updating never mutates the previous state', () => {
  const { values } = repSignal({ reps: 2, repSeconds: 1 });
  const before = countSignal(values.slice(0, 30));
  const snapshot = JSON.stringify(before);
  updateSignalCounter(before, 0.4, 5000);
  assert.equal(JSON.stringify(before), snapshot);
  assert.ok(Object.isFrozen(before));
});

test('resetting the zones forgets home but keeps the count and its timing', () => {
  const counted = countSignal(repSignal({ reps: 3, repSeconds: 1 }).values);
  const reset = resetZones(counted);
  assert.equal(reset.count, 3);
  assert.deepEqual(reset.repTimes, counted.repTimes);
  assert.equal(reset.home, null);
  assert.equal(reset.zone, null);
});

test('a home first chosen on jitter is reconsidered once the real movement dwarfs it', () => {
  // Regression from a randomized sweep: rest jitter just over the PCA stillness floor fixed "home" on
  // the wrong side, and the single real squat that followed was never counted.
  const stream = poseStream({ segments: [{ reps: 1, repSeconds: 2.25 }], seed: 139, jitter: 0.0038 });
  assert.equal(countStream(stream, { signal: SIGNAL.PCA }).count, 1);
});

test('once a rep has confirmed home, a growing range mid-set no longer moves it', () => {
  // Regression from the same sweep: home was re-read mid-set from a window head that was mid-rep.
  const { values } = repSignal({ reps: 10, repSeconds: 2.97, amplitude: 0.91, noise: 0.095, seed: 45 });
  assert.equal(countSignal(values, { minAmplitude: 0.38 }).count, 10);
});

test('a window too small to show movement leaves the state untouched', () => {
  const state = createSignalCounter({ minAmplitude: 0.1 });
  assert.equal(stepWindow(state, [0.5, 0.52, 0.49], 100), state);
});

test('a shallow bounce soon after a rep is held back by the period gap', () => {
  // Five 1 s reps establish the period; then a quick 0.3 s bounce that only just crosses the cut.
  const { values } = repSignal({ reps: 5, repSeconds: 1, amplitude: 1, restSeconds: 1 });
  const head = values.slice(0, -FPS);
  const bounce = [0.15, 0.4, 0.75, 0.75, 0.4, 0.1, 0, 0].map((v) => v);
  const counted = countSignal([...head, ...bounce, ...new Array(FPS).fill(0)], { minAmplitude: 0.1, smoothing: 1 });
  assert.equal(counted.count, 5);
});

// ---------------------------------------------------------------- signal choice

test('the principal component follows the direction of motion, with a stable sign', () => {
  const window = Array.from({ length: 40 }, (_, i) => Float64Array.from([Math.sin(i / 3), 0.01 * Math.cos(i), 0]));
  const first = principalComponent(window, null);
  assert.ok(Math.abs(first[0]) > 0.99);
  const flipped = principalComponent(window, first.map((v) => -v));
  assert.ok(flipped[0] * -first[0] > 0, 'aligned with the previous vector, not flipped back');
});

test('the busiest angle switches only when another joint clearly moves more', () => {
  const window = (moving, amount) => Array.from({ length: 30 }, (_, i) => {
    const row = new Float64Array(34);
    row[26 + moving] = amount * Math.sin(i / 2);
    row[26 + 6] = 0.1 * Math.sin(i / 2);
    return row;
  });
  assert.equal(busiestAngle(window(2, 1), null), 2);
  assert.equal(busiestAngle(window(2, 0.11), 6), 6, '0.11 vs 0.1 is not a clear win');
  assert.equal(busiestAngle(window(2, 1), 6), 2);
});

// ---------------------------------------------------------------- frame level

test('synthetic squats are counted with both signal choices', () => {
  for (const signal of [SIGNAL.PCA, SIGNAL.ANGLE]) {
    for (const reps of [1, 3, 6]) {
      const stream = poseStream({ segments: [{ reps, repSeconds: 2 }] });
      assert.equal(countStream(stream, { signal }).count, reps, `${signal}: ${reps} squats`);
    }
  }
});

test('an arms-only movement is counted too: nothing is squat-specific', () => {
  for (const signal of [SIGNAL.PCA, SIGNAL.ANGLE]) {
    const stream = poseStream({ segments: [{ reps: 5, repSeconds: 1.5 }], pose: curlPose });
    assert.equal(countStream(stream, { signal }).count, 5, signal);
  }
});

test('a set that speeds up is counted in full', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 2.5 }, { reps: 4, repSeconds: 1.2 }] });
  assert.equal(countStream(stream).count, 7);
});

test('rep times line up with the synthetic ground truth', () => {
  const stream = poseStream({ segments: [{ reps: 4, repSeconds: 2 }] });
  const { repTimes } = countStream(stream);
  repTimes.forEach((t, i) => assert.ok(Math.abs(t - stream.repTimesMs[i]) <= 500,
    `rep ${i}: ${t} vs ${stream.repTimesMs[i]}`));
});

test('standing still with tracking jitter counts nothing', () => {
  const stream = poseStream({ segments: [], restSeconds: 8, jitter: 0.004 });
  assert.equal(countStream(stream).count, 0);
  assert.equal(countStream(stream, { signal: SIGNAL.ANGLE }).count, 0);
});

test('frames without a pose are survived: the last pose is held', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 2 }] });
  const frames = stream.frames.map((frame, i) => (i % 10 === 5 ? null : frame));
  assert.equal(countStream({ ...stream, frames }).count, 3);
  assert.equal(countStream({ ...stream, frames: stream.frames.map(() => undefined) }).count, 0);
});

test('a frame rate other than 15 fps is resampled, including a jump in time', () => {
  const stream = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], fps: 60 });
  assert.equal(countStream(stream).count, 3);
  const jumped = updateGenericCounter(countStream(stream), stream.frames[0], stream.timestampsMs.at(-1) + 60_000,
    stream.size);
  assert.ok(Number.isFinite(jumped.nextSampleMs), 'a one-minute gap does not loop forever');
});

test('frame updates never mutate the previous state', () => {
  const stream = poseStream({ segments: [{ reps: 1, repSeconds: 1 }] });
  const before = countStream({ ...stream, frames: stream.frames.slice(0, 40), timestampsMs: stream.timestampsMs.slice(0, 40) });
  const count = before.count;
  const windowLength = before.window.length;
  updateGenericCounter(before, stream.frames[41], stream.timestampsMs[41], stream.size);
  assert.equal(before.count, count);
  assert.equal(before.window.length, windowLength);
  assert.ok(Object.isFrozen(before));
});

test('the counter reports movement while reps happen and stillness at rest', () => {
  const moving = poseStream({ segments: [{ reps: 3, repSeconds: 2 }], restSeconds: 0, tailSeconds: 0 });
  assert.equal(countStream(moving).moving, true);
  const still = poseStream({ segments: [], restSeconds: 6, jitter: 0.002 });
  assert.equal(countStream(still).moving, false);
  assert.equal(createGenericCounter().moving, false);
});
