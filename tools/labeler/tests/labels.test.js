import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  REP_LABEL_COLUMNS, SPEEDS, MIN_MARK_GAP_MS, STATUS, SESSION,
  parseCsv, toCsv, videoIdFromFilename, pickVideoFiles, nextSpeed, frameStepMs, recheckSubset,
  emptyLabel, addMark, removeLastMark, setHold, clearLastHoldEdge, setAmbiguous, sortedMarks,
  labelStatus, labelProblems, toRow, fromRow, exportRows, nextTodo, storageKey,
} from '../src/labels.js';

const NOW = '2026-10-02T09:15:00.000Z';
const SQUAT = { video_id: 'squat_1', label: 'squat', task: 'reps', fps: '30', duration_s: '10' };
const PLANK = { video_id: 'plank_1', label: 'plank', task: 'hold', fps: '30', duration_s: '40' };

const marked = (entry, ...times) => times.reduce((label, ms) => addMark(label, ms, NOW).label, emptyLabel(entry));

// ---------------------------------------------------------------- CSV

test('a to_label.csv row parses into an object keyed by the header', () => {
  const rows = parseCsv('video_id,label,task\nsquat_1,squat,reps\nplank_1,plank,hold\n');
  assert.deepEqual(rows, [
    { video_id: 'squat_1', label: 'squat', task: 'reps' },
    { video_id: 'plank_1', label: 'plank', task: 'hold' },
  ]);
});

test('quoted fields keep their commas, quotes and line breaks', () => {
  const [row] = parseCsv('id,notes\nv1,"dua orang, satu ""asing""\nbaris kedua"\n');
  assert.equal(row.notes, 'dua orang, satu "asing"\nbaris kedua');
});

test('Windows line endings, a missing final newline and blank lines are all tolerated', () => {
  assert.deepEqual(parseCsv('a,b\r\n1,2\r\n\r\n3,4'), [{ a: '1', b: '2' }, { a: '3', b: '4' }]);
});

test('empty cells stay empty strings and short rows are padded', () => {
  assert.deepEqual(parseCsv('a,b,c\n1,,\n2\n'), [{ a: '1', b: '', c: '' }, { a: '2', b: '', c: '' }]);
});

test('an empty file parses to no rows', () => {
  assert.deepEqual(parseCsv(''), []);
  assert.deepEqual(parseCsv('a,b\n'), []);
});

test('toCsv quotes only what needs quoting and round-trips through parseCsv', () => {
  const rows = [{ a: 'plain', b: 'with, comma' }, { a: 'say "hi"', b: '' }];
  const text = toCsv(rows, ['a', 'b']);
  assert.equal(text, 'a,b\nplain,"with, comma"\n"say ""hi""",\n');
  assert.deepEqual(parseCsv(text), rows);
});

// ---------------------------------------------------------------- helpers

test('video ids follow the same rule as the Python manifest', () => {
  // Cases mirror ml/tests/data/test_manifest.py so a picked file matches its to_label.csv row.
  assert.equal(videoIdFromFilename('push-up_17.mp4'), 'push_up_17');
  assert.equal(videoIdFromFilename('workout-videos/push-up/push-up_17.mp4'), 'push_up_17');
  assert.equal(videoIdFromFilename('C:\\data\\pull Up\\Pull Up 3.MOV'), 'pull_up_3');
  assert.equal(videoIdFromFilename('  tricep Pushdown_12.mov '), 'tricep_pushdown_12');
  assert.equal(videoIdFromFilename('a.b.mp4'), 'a_b');
  assert.equal(videoIdFromFilename('-weird--name-.mp4'), 'weird_name');
});

test('a picked folder maps each video id to one playable file', () => {
  const files = ['push-up/push-up_17.mp4', 'plank/Plank 1.MOV', 'notes.txt', 'squat/.DS_Store']
    .map((name) => ({ name: name.split('/').pop() }));
  const picked = pickVideoFiles(files);
  assert.deepEqual([...picked.keys()].sort(), ['plank_1', 'push_up_17']);
});

test('a converted .mp4 beats the HEVC .MOV it came from, whichever is listed first', () => {
  for (const order of [['IMG_1.MOV', 'IMG_1.mp4'], ['IMG_1.mp4', 'IMG_1.MOV']]) {
    assert.equal(pickVideoFiles(order.map((name) => ({ name }))).get('img_1').name, 'IMG_1.mp4');
  }
});

test('speed steps through 0.5×, 1× and 2× and stops at the ends', () => {
  assert.deepEqual(SPEEDS, [0.5, 1, 2]);
  assert.equal(nextSpeed(1, -1), 0.5);
  assert.equal(nextSpeed(0.5, -1), 0.5);
  assert.equal(nextSpeed(1, 1), 2);
  assert.equal(nextSpeed(2, 1), 2);
  assert.equal(nextSpeed(1.75, 1), 2, 'an unknown speed restarts from 1×');
});

test('one frame step uses the video fps and falls back to 30', () => {
  assert.ok(Math.abs(frameStepMs('29.97') - 33.367) < 0.001);
  assert.ok(Math.abs(frameStepMs('') - 33.333) < 0.001);
  assert.ok(Math.abs(frameStepMs(0) - 33.333) < 0.001);
});

test('the recheck subset is about ten percent, deterministic and order-independent', () => {
  const entries = Array.from({ length: 104 }, (_, i) => ({ video_id: `video_${i}` }));
  const subset = recheckSubset(entries);
  assert.equal(subset.length, 11);
  assert.deepEqual(recheckSubset([...entries].reverse()).map((e) => e.video_id).sort(),
    subset.map((e) => e.video_id).sort());
  assert.deepEqual(recheckSubset(entries), subset);
});

test('the recheck subset keeps at least one video and handles an empty list', () => {
  assert.equal(recheckSubset([{ video_id: 'only_1' }]).length, 1);
  assert.deepEqual(recheckSubset([]), []);
});

test('each session stores under its own key so a recheck never sees the first labels', () => {
  assert.notEqual(storageKey(SESSION.MAIN), storageKey(SESSION.RECHECK));
});

// ---------------------------------------------------------------- marks

test('a blank label knows its task from to_label.csv', () => {
  assert.equal(emptyLabel(SQUAT).task, 'reps');
  assert.equal(emptyLabel(PLANK).task, 'hold');
  assert.equal(emptyLabel({ video_id: 'x', label: 'x' }).task, 'reps');
});

test('marking a rep rounds to whole milliseconds and records when', () => {
  const { label, added } = addMark(emptyLabel(SQUAT), 1234.6, NOW);
  assert.equal(added, true);
  assert.deepEqual(label.marks, [1235]);
  assert.equal(label.updatedAt, NOW);
});

test('a second press within the minimum gap is rejected rather than counted twice', () => {
  const first = marked(SQUAT, 2000);
  const { label, added } = addMark(first, 2000 + MIN_MARK_GAP_MS - 1, NOW);
  assert.equal(added, false);
  assert.equal(label, first, 'the label is returned unchanged');
  assert.equal(addMark(first, 2000 + MIN_MARK_GAP_MS, NOW).added, true);
});

test('backspace undoes the latest press even after seeking backwards', () => {
  const label = marked(SQUAT, 5000, 2000);
  assert.deepEqual(removeLastMark(label, NOW).marks, [5000]);
  assert.deepEqual(sortedMarks(label), [2000, 5000]);
});

test('undo on an empty label changes nothing', () => {
  const blank = emptyLabel(SQUAT);
  assert.equal(removeLastMark(blank), blank);
});

test('marking never mutates the label it was given', () => {
  const before = marked(SQUAT, 1000);
  const snapshot = JSON.stringify(before);
  addMark(before, 4000, NOW);
  removeLastMark(before, NOW);
  setAmbiguous(before, true, 'x', NOW);
  assert.equal(JSON.stringify(before), snapshot);
});

test('holds set their start and end, and undo clears the end before the start', () => {
  const held = setHold(setHold(emptyLabel(PLANK), 'start', 1500.4, NOW), 'end', 38000, NOW);
  assert.equal(held.holdStart, 1500);
  assert.equal(held.holdEnd, 38000);
  const once = clearLastHoldEdge(held, NOW);
  assert.equal(once.holdEnd, null);
  assert.equal(once.holdStart, 1500);
  assert.equal(clearLastHoldEdge(clearLastHoldEdge(once, NOW), NOW).holdStart, null);
});

// ---------------------------------------------------------------- status and problems

test('status follows the task: reps need a mark, holds need both edges, ambiguous wins', () => {
  assert.equal(labelStatus(undefined), STATUS.TODO);
  assert.equal(labelStatus(emptyLabel(SQUAT)), STATUS.TODO);
  assert.equal(labelStatus(marked(SQUAT, 1000)), STATUS.DONE);
  assert.equal(labelStatus(setHold(emptyLabel(PLANK), 'start', 0, NOW)), STATUS.TODO);
  assert.equal(labelStatus(setHold(setHold(emptyLabel(PLANK), 'start', 0, NOW), 'end', 9, NOW)), STATUS.DONE);
  assert.equal(labelStatus(setAmbiguous(emptyLabel(SQUAT), true, 'bukan squat', NOW)), STATUS.AMBIGUOUS);
});

test('problems the validator would reject are caught before export', () => {
  assert.deepEqual(labelProblems(marked(SQUAT, 1000), 10000), []);
  assert.deepEqual(labelProblems(setAmbiguous(emptyLabel(SQUAT), true, ' ', NOW), 10000),
    ['Ditandai ambigu tanpa catatan']);
  assert.deepEqual(labelProblems(marked(SQUAT, 12000), 10000), ['Ada tanda rep melewati durasi video']);
  const backwards = setHold(setHold(emptyLabel(PLANK), 'start', 9000, NOW), 'end', 3000, NOW);
  assert.deepEqual(labelProblems(backwards, 40000), ['Mulai tahan harus sebelum selesai']);
  const long = setHold(emptyLabel(PLANK), 'end', 50000, NOW);
  assert.deepEqual(labelProblems(long, 40000), ['Waktu tahan melewati durasi video']);
  assert.deepEqual(labelProblems(marked(SQUAT, 99999), NaN), [], 'unknown duration skips the range check');
});

// ---------------------------------------------------------------- rows

test('a rep label becomes a rep_labels.csv row with sorted marks', () => {
  assert.deepEqual(toRow(marked(SQUAT, 4500, 2000, 7000), 'gk'), {
    video_id: 'squat_1', label: 'squat', rep_count: '3', rep_timestamps_ms: '2000;4500;7000',
    hold_start_ms: '', hold_end_ms: '', is_ambiguous: 'false', notes: '', labeler: 'gk', labeled_at: NOW,
  });
});

test('a hold label leaves the rep columns empty', () => {
  const held = setHold(setHold(emptyLabel(PLANK), 'start', 1500, NOW), 'end', 38000, NOW);
  const row = toRow(held, 'gk');
  assert.equal(row.rep_count, '');
  assert.equal(row.rep_timestamps_ms, '');
  assert.equal(row.hold_start_ms, '1500');
  assert.equal(row.hold_end_ms, '38000');
});

test('rows carry every column the Python validator expects', () => {
  assert.deepEqual(Object.keys(toRow(emptyLabel(SQUAT), 'gk')), [...REP_LABEL_COLUMNS]);
});

test('an exported label resumes unchanged after a CSV round trip', () => {
  const original = setAmbiguous(marked(SQUAT, 2000, 4500), true, 'sudut kamera, "aneh"', NOW);
  const [row] = parseCsv(toCsv([toRow(original, 'gk')], REP_LABEL_COLUMNS));
  assert.deepEqual(fromRow(row, 'reps'), { ...original, marks: sortedMarks(original) });

  const held = setHold(setHold(emptyLabel(PLANK), 'start', 0, NOW), 'end', 38000, NOW);
  const [holdRow] = parseCsv(toCsv([toRow(held, 'gk')], REP_LABEL_COLUMNS));
  assert.deepEqual(fromRow(holdRow, 'hold'), held);
});

test('only finished or ambiguous labels are exported, in to_label.csv order', () => {
  const entries = [SQUAT, { ...SQUAT, video_id: 'squat_2' }, PLANK];
  const labels = {
    plank_1: setAmbiguous(emptyLabel(PLANK), true, 'tidak terlihat', NOW),
    squat_2: emptyLabel(entries[1]),
    squat_1: marked(SQUAT, 1000),
  };
  assert.deepEqual(exportRows(entries, labels, 'gk').map((r) => r.video_id), ['squat_1', 'plank_1']);
});

test('next jumps to the following unfinished video and wraps around', () => {
  const entries = [SQUAT, { ...SQUAT, video_id: 'squat_2' }, PLANK];
  const labels = { squat_2: marked(entries[1], 1000) };
  assert.equal(nextTodo(entries, labels, 0), 2);
  assert.equal(nextTodo(entries, labels, 2), 0);
  const allDone = { squat_1: marked(SQUAT, 1), squat_2: marked(entries[1], 1),
    plank_1: setAmbiguous(emptyLabel(PLANK), true, 'x', NOW) };
  assert.equal(nextTodo(entries, allDone, 0), -1);
});
