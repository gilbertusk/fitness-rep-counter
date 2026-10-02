/**
 * Pure logic for the rep labeler: CSV in and out, the per-video label state, and the small rules
 * from `labels/README.md`. No DOM and no storage here — `app.js` owns those.
 *
 * Every function returns a new value; a label passed in is never modified.
 */

// Must match REP_LABEL_COLUMNS in ml/src/repcount/labels/validate.py.
export const REP_LABEL_COLUMNS = Object.freeze([
  'video_id', 'label', 'rep_count', 'rep_timestamps_ms', 'hold_start_ms', 'hold_end_ms',
  'is_ambiguous', 'notes', 'labeler', 'labeled_at',
]);

export const SPEEDS = Object.freeze([0.5, 1, 2]);
// Two marks closer than this are one key pressed twice, not two reps.
export const MIN_MARK_GAP_MS = 150;
export const RECHECK_FRACTION = 0.1;
const DEFAULT_FPS = 30;

export const STATUS = Object.freeze({ TODO: 'todo', DONE: 'done', AMBIGUOUS: 'ambiguous' });
export const SESSION = Object.freeze({ MAIN: 'utama', RECHECK: 'cek-ulang' });

// ---------------------------------------------------------------- CSV

/** RFC 4180 CSV → array of row objects keyed by the header. Handles quotes, commas and CRLF. */
export function parseCsv(text) {
  const records = [];
  let record = [];
  let field = '';
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { record.push(field); field = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      record.push(field);
      records.push(record);
      record = [];
      field = '';
    } else field += char;
  }
  if (field !== '' || record.length) records.push([...record, field]);

  const [header = [], ...rows] = records.filter((r) => r.some((value) => value !== ''));
  return rows.map((values) => Object.fromEntries(header.map((name, i) => [name.trim(), values[i] ?? ''])));
}

function quote(value) {
  const text = String(value ?? '');
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Row objects → CSV text with a header line, in the given column order. */
export function toCsv(rows, columns) {
  return [columns.join(','), ...rows.map((row) => columns.map((c) => quote(row[c])).join(','))].join('\n') + '\n';
}

// ---------------------------------------------------------------- small helpers

/** Same rule as repcount.data.manifest.make_video_id: 'push-up_17.mp4' → 'push_up_17'. */
export function videoIdFromFilename(name) {
  const base = String(name).split(/[\\/]/).pop().trim();
  const stem = base.includes('.') ? base.slice(0, base.lastIndexOf('.')) : base;
  return stem.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

// When a folder holds the same clip twice (an HEVC .MOV next to the .mp4 converted from it, as the
// labeler's playback-error message suggests), the most widely playable container wins.
const EXTENSION_PREFERENCE = ['mp4', 'webm', 'm4v', 'mov'];

/**
 * Picked files → Map of video_id → the file to play, skipping non-video files.
 * @param {Array<{name: string}>} files e.g. the FileList of a folder picker
 */
export function pickVideoFiles(files) {
  const rank = (name) => EXTENSION_PREFERENCE.indexOf(name.split('.').pop().toLowerCase());
  const chosen = new Map();
  for (const file of files) {
    if (rank(file.name) === -1) continue;
    const id = videoIdFromFilename(file.name);
    const current = chosen.get(id);
    if (!current || rank(file.name) < rank(current.name)) chosen.set(id, file);
  }
  return chosen;
}

/** One step through SPEEDS, clamped at both ends. */
export function nextSpeed(current, direction) {
  const index = SPEEDS.indexOf(current);
  const from = index === -1 ? SPEEDS.indexOf(1) : index;
  return SPEEDS[Math.min(SPEEDS.length - 1, Math.max(0, from + Math.sign(direction)))];
}

export function frameStepMs(fps) {
  const rate = Number(fps);
  return 1000 / (Number.isFinite(rate) && rate > 0 ? rate : DEFAULT_FPS);
}

/** FNV-1a: a tiny stable hash, so the recheck subset is the same on every machine. */
function hash(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

/** The ±10% of entries to label a second time, chosen deterministically from their video ids. */
export function recheckSubset(entries, fraction = RECHECK_FRACTION) {
  if (!entries.length) return [];
  const n = Math.max(1, Math.ceil(entries.length * fraction));
  const chosen = new Set([...entries].sort((a, b) => hash(a.video_id) - hash(b.video_id)
    || a.video_id.localeCompare(b.video_id)).slice(0, n).map((e) => e.video_id));
  return entries.filter((entry) => chosen.has(entry.video_id));
}

// ---------------------------------------------------------------- label state

/** A blank label for one row of to_label.csv. */
export function emptyLabel(entry) {
  return {
    videoId: entry.video_id,
    label: entry.label,
    task: entry.task === 'hold' ? 'hold' : 'reps',
    marks: [],
    holdStart: null,
    holdEnd: null,
    ambiguous: false,
    notes: '',
    updatedAt: null,
  };
}

const touched = (label, changes, now) => ({ ...label, ...changes, updatedAt: now });

/**
 * Mark one rep at `ms`. Marks keep the order they were added in, so Backspace undoes the latest
 * key press even after seeking backwards; they are sorted on export.
 * @returns {{ label: object, added: boolean }} added is false for a mark too close to another
 */
export function addMark(label, ms, now = new Date().toISOString()) {
  const at = Math.max(0, Math.round(ms));
  if (label.marks.some((mark) => Math.abs(mark - at) < MIN_MARK_GAP_MS)) return { label, added: false };
  return { label: touched(label, { marks: [...label.marks, at] }, now), added: true };
}

export function removeLastMark(label, now = new Date().toISOString()) {
  if (!label.marks.length) return label;
  return touched(label, { marks: label.marks.slice(0, -1) }, now);
}

/** Set the start or end of a plank hold. */
export function setHold(label, edge, ms, now = new Date().toISOString()) {
  const key = edge === 'start' ? 'holdStart' : 'holdEnd';
  return touched(label, { [key]: Math.max(0, Math.round(ms)) }, now);
}

/** Undo for holds: clear the end first, then the start. */
export function clearLastHoldEdge(label, now = new Date().toISOString()) {
  if (label.holdEnd !== null) return touched(label, { holdEnd: null }, now);
  if (label.holdStart !== null) return touched(label, { holdStart: null }, now);
  return label;
}

export function setAmbiguous(label, ambiguous, notes = label.notes, now = new Date().toISOString()) {
  return touched(label, { ambiguous: Boolean(ambiguous), notes: String(notes) }, now);
}

export const sortedMarks = (label) => [...label.marks].sort((a, b) => a - b);

export function labelStatus(label) {
  if (!label) return STATUS.TODO;
  if (label.ambiguous) return STATUS.AMBIGUOUS;
  const finished = label.task === 'hold' ? label.holdStart !== null && label.holdEnd !== null : label.marks.length > 0;
  return finished ? STATUS.DONE : STATUS.TODO;
}

/**
 * What would make `python -m repcount.labels.validate` reject this label, checked before export so
 * the labeler can fix it while the video is still open.
 */
export function labelProblems(label, durationMs) {
  const problems = [];
  const limit = Number.isFinite(durationMs) ? durationMs : Infinity;
  if (label.ambiguous && !label.notes.trim()) problems.push('Ditandai ambigu tanpa catatan');
  if (label.task === 'hold') {
    const { holdStart: start, holdEnd: end } = label;
    if (start !== null && end !== null && !(start < end)) problems.push('Mulai tahan harus sebelum selesai');
    if ([start, end].some((t) => t !== null && t > limit)) problems.push('Waktu tahan melewati durasi video');
  } else if (label.marks.some((t) => t > limit)) {
    problems.push('Ada tanda rep melewati durasi video');
  }
  return problems;
}

// ---------------------------------------------------------------- rows in and out

/** A label → one row of rep_labels.csv (see labels/README.md for the format). */
export function toRow(label, labeler) {
  const hold = label.task === 'hold';
  const marks = sortedMarks(label);
  return {
    video_id: label.videoId,
    label: label.label,
    rep_count: hold ? '' : String(marks.length),
    rep_timestamps_ms: hold ? '' : marks.join(';'),
    hold_start_ms: hold && label.holdStart !== null ? String(label.holdStart) : '',
    hold_end_ms: hold && label.holdEnd !== null ? String(label.holdEnd) : '',
    is_ambiguous: label.ambiguous ? 'true' : 'false',
    notes: label.notes,
    labeler,
    labeled_at: label.updatedAt ?? '',
  };
}

const intOrNull = (text) => (String(text ?? '').trim() === '' ? null : Number.parseInt(text, 10));

/** One row of a previously exported rep_labels.csv → a label, to resume work. */
export function fromRow(row, task) {
  return {
    videoId: row.video_id,
    label: row.label,
    task: task === 'hold' ? 'hold' : 'reps',
    marks: String(row.rep_timestamps_ms ?? '').split(';').filter((t) => t.trim() !== '').map(Number),
    holdStart: intOrNull(row.hold_start_ms),
    holdEnd: intOrNull(row.hold_end_ms),
    ambiguous: row.is_ambiguous === 'true',
    notes: row.notes ?? '',
    updatedAt: row.labeled_at || null,
  };
}

/** The rows to export: finished or ambiguous labels only, in the order of to_label.csv. */
export function exportRows(entries, labels, labeler) {
  return entries
    .map((entry) => labels[entry.video_id])
    .filter((label) => labelStatus(label) !== STATUS.TODO)
    .map((label) => toRow(label, labeler));
}

/** Index of the next unfinished entry after `from`, wrapping around; -1 when everything is done. */
export function nextTodo(entries, labels, from) {
  for (let step = 1; step <= entries.length; step += 1) {
    const index = (from + step) % entries.length;
    if (labelStatus(labels[entries[index].video_id]) === STATUS.TODO) return index;
  }
  return -1;
}

export const storageKey = (session) => `repLabeler:v1:${session}`;
