import {
  REP_LABEL_COLUMNS, STATUS, SESSION,
  parseCsv, toCsv, pickVideoFiles, nextSpeed, frameStepMs, recheckSubset,
  emptyLabel, addMark, removeLastMark, setHold, clearLastHoldEdge, setAmbiguous, sortedMarks,
  labelStatus, labelProblems, fromRow, exportRows, nextTodo, storageKey,
} from './labels.js';

/** DOM and storage for the labeler. Every rule lives in labels.js; this file only wires it up. */

const $ = (id) => document.getElementById(id);
const ui = {
  toLabelFile: $('to-label-file'), videoFolder: $('video-folder'), importFile: $('import-file'),
  labeler: $('labeler'), recheck: $('recheck-mode'), exportButton: $('export'), banner: $('recheck-banner'),
  progress: $('progress'), onlyTodo: $('only-todo'), list: $('video-list'), emptyHint: $('empty-hint'),
  title: $('video-title'), badge: $('task-badge'), video: $('video'), videoError: $('video-error'),
  timeline: $('timeline'), holdSpan: $('hold-span'), playhead: $('playhead'),
  time: $('time'), countWrap: $('count-wrap'), count: $('count'), holdWrap: $('hold-wrap'), hold: $('hold'),
  speed: $('speed'), ambiguous: $('ambiguous'), notes: $('notes'), status: $('status'), problems: $('problems'),
};

const PREFS_KEY = 'repLabeler:v1:prefs';
const EXPORT_NAMES = { [SESSION.MAIN]: 'rep_labels.csv', [SESSION.RECHECK]: 'rep_labels_recheck.csv' };

const state = {
  allEntries: [],
  files: new Map(),
  session: SESSION.MAIN,
  labels: {},
  index: -1,
  speed: 1,
};
let objectUrl = null;

// ---------------------------------------------------------------- storage (may be unavailable)

function readStore(key) {
  try {
    return JSON.parse(localStorage.getItem(key) ?? 'null');
  } catch {
    return null;
  }
}

function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function saveSession() {
  if (!writeStore(storageKey(state.session), { labels: state.labels })) {
    setStatus('Penyimpanan otomatis tidak tersedia di browser ini — Export CSV secara berkala.', true);
  }
}

const loadSession = () => { state.labels = readStore(storageKey(state.session))?.labels ?? {}; };
const savePrefs = () => writeStore(PREFS_KEY, { labeler: ui.labeler.value, speed: state.speed, entries: state.allEntries });

// ---------------------------------------------------------------- current video

const entries = () => (state.session === SESSION.RECHECK ? recheckSubset(state.allEntries) : state.allEntries);
const currentEntry = () => entries()[state.index];
const currentLabel = () => {
  const entry = currentEntry();
  return entry ? state.labels[entry.video_id] ?? emptyLabel(entry) : null;
};
const durationMs = () => (Number.isFinite(ui.video.duration) ? ui.video.duration * 1000
  : Number(currentEntry()?.duration_s) * 1000);

function setStatus(message, warn = false) {
  ui.status.textContent = message;
  ui.status.style.color = warn ? 'var(--mark)' : '';
}

function updateLabel(change) {
  const label = currentLabel();
  if (!label) return;
  const next = change(label);
  if (next === label) return;
  state.labels = { ...state.labels, [label.videoId]: next };
  saveSession();
  render();
}

function openEntry(index) {
  const entry = entries()[index];
  if (!entry) return;
  state.index = index;
  const file = state.files.get(entry.video_id);
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = file ? URL.createObjectURL(file) : null;
  ui.videoError.hidden = Boolean(file);
  ui.videoError.textContent = file ? '' : 'Berkas video tidak ditemukan — pilih folder data/workout-videos/.';
  if (file) ui.video.src = objectUrl;
  else ui.video.removeAttribute('src');
  ui.video.load();
  setStatus('');
  render();
  ui.list.querySelector('li.current')?.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------- actions (one per shortcut)

const togglePlay = () => (ui.video.paused ? ui.video.play().catch(() => {}) : ui.video.pause());

function stepFrame(direction) {
  ui.video.pause();
  const step = frameStepMs(currentEntry()?.fps) / 1000;
  const end = Number.isFinite(ui.video.duration) ? ui.video.duration : Infinity;
  ui.video.currentTime = Math.min(end, Math.max(0, ui.video.currentTime + direction * step));
}

function changeSpeed(direction) {
  state.speed = nextSpeed(state.speed, direction);
  ui.video.playbackRate = state.speed;
  ui.video.defaultPlaybackRate = state.speed;
  savePrefs();
  renderReadout();
}

function markRep() {
  const label = currentLabel();
  if (!label) return;
  if (label.task === 'hold') { setStatus('Plank dilabel sebagai durasi tahan: pakai S dan E.', true); return; }
  const { label: next, added } = addMark(label, ui.video.currentTime * 1000);
  if (!added) { setStatus('Terlalu dekat dengan tanda lain — diabaikan.', true); return; }
  updateLabel(() => next);
  setStatus(`Rep ${next.marks.length} ditandai di ${(ui.video.currentTime).toFixed(2)} s`);
}

function markHold(edge) {
  const label = currentLabel();
  if (!label) return;
  if (label.task !== 'hold') { setStatus('S/E hanya untuk plank. Untuk repetisi pakai R.', true); return; }
  updateLabel((l) => setHold(l, edge, ui.video.currentTime * 1000));
}

const undo = () => updateLabel((l) => (l.task === 'hold' ? clearLastHoldEdge(l) : removeLastMark(l)));

function toggleAmbiguous() {
  updateLabel((l) => setAmbiguous(l, !l.ambiguous));
  if (currentLabel()?.ambiguous) ui.notes.focus();
}

function goNext() {
  const index = nextTodo(entries(), state.labels, state.index);
  if (index === -1) setStatus('Semua video sudah dilabel — saatnya Export CSV.');
  else openEntry(index);
}

const KEYS = {
  ' ': togglePlay, ArrowLeft: () => stepFrame(-1), ArrowRight: () => stepFrame(1),
  ',': () => changeSpeed(-1), '.': () => changeSpeed(1), r: markRep, Backspace: undo,
  s: () => markHold('start'), e: () => markHold('end'), f: toggleAmbiguous, n: goNext,
};

function onKey(event) {
  if (event.target.matches('input[type="text"], textarea')) {
    if (event.key === 'Escape') event.target.blur();
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const action = KEYS[event.key.length === 1 ? event.key.toLowerCase() : event.key];
  if (!action) return;
  event.preventDefault(); // also stops Space from clicking a focused button or checkbox
  action();
}

// ---------------------------------------------------------------- rendering

function listItem(entry, index) {
  const label = state.labels[entry.video_id];
  const status = labelStatus(label);
  const item = document.createElement('li');
  item.dataset.status = status;
  item.classList.toggle('current', index === state.index);
  item.classList.toggle('missing', state.files.size > 0 && !state.files.has(entry.video_id));
  const count = label && entry.task !== 'hold' ? `${label.marks.length} rep` : '';
  const mark = { [STATUS.DONE]: '✓', [STATUS.AMBIGUOUS]: '?', [STATUS.TODO]: '·' }[status];
  item.innerHTML = '<span class="name"></span><span class="meta"></span>';
  item.querySelector('.name').textContent = entry.video_id;
  item.querySelector('.meta').textContent = `${count} ${mark}`.trim();
  item.addEventListener('click', () => openEntry(index));
  return item;
}

function renderList() {
  const list = entries();
  const items = [];
  let group = null;
  list.forEach((entry, index) => {
    if (ui.onlyTodo.checked && labelStatus(state.labels[entry.video_id]) !== STATUS.TODO && index !== state.index) return;
    if (entry.label !== group) {
      group = entry.label;
      const header = document.createElement('li');
      header.className = 'group';
      header.textContent = group;
      items.push(header);
    }
    items.push(listItem(entry, index));
  });
  ui.list.replaceChildren(...items);
  ui.emptyHint.hidden = list.length > 0 && state.files.size > 0;
  const finished = list.filter((e) => labelStatus(state.labels[e.video_id]) !== STATUS.TODO).length;
  ui.progress.textContent = `${finished} / ${list.length}`;
}

function renderTimeline(label) {
  ui.timeline.querySelectorAll('.tick').forEach((tick) => tick.remove());
  const total = durationMs();
  const at = (ms) => `${Math.min(100, (ms / total) * 100)}%`;
  ui.holdSpan.hidden = !(label?.task === 'hold' && label.holdStart !== null && Number.isFinite(total));
  if (!label || !Number.isFinite(total) || total <= 0) return;
  if (label.task === 'hold') {
    const end = label.holdEnd ?? ui.video.currentTime * 1000;
    ui.holdSpan.style.left = at(label.holdStart ?? 0);
    ui.holdSpan.style.width = `calc(${at(Math.max(0, end - (label.holdStart ?? 0)))})`;
    return;
  }
  sortedMarks(label).forEach((ms) => {
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = at(ms);
    ui.timeline.append(tick);
  });
}

function renderReadout() {
  const total = durationMs();
  ui.time.textContent = `${ui.video.currentTime.toFixed(2)} / ${Number.isFinite(total) ? (total / 1000).toFixed(2) : '–'} s`;
  ui.playhead.style.left = Number.isFinite(total) && total > 0 ? `${(ui.video.currentTime * 1000 / total) * 100}%` : '0';
  ui.speed.textContent = `${state.speed}×`;
}

function renderCurrent() {
  const entry = currentEntry();
  const label = currentLabel();
  ui.title.textContent = entry ? `${entry.video_id} — ${entry.label}` : 'Belum ada video';
  ui.badge.hidden = !entry;
  ui.badge.textContent = entry?.task === 'hold' ? 'tahan (S/E)' : 'repetisi (R)';
  ui.countWrap.hidden = entry?.task === 'hold';
  ui.holdWrap.hidden = entry?.task !== 'hold';
  ui.count.textContent = label ? String(label.marks.length) : '0';
  const seconds = (ms) => (ms === null ? '–' : (ms / 1000).toFixed(2));
  ui.hold.textContent = label ? `${seconds(label.holdStart)} → ${seconds(label.holdEnd)} s` : '–';
  ui.ambiguous.checked = Boolean(label?.ambiguous);
  if (document.activeElement !== ui.notes) ui.notes.value = label?.notes ?? '';
  ui.problems.replaceChildren(...(label ? labelProblems(label, durationMs()) : []).map((problem) => {
    const item = document.createElement('li');
    item.textContent = problem;
    return item;
  }));
  renderTimeline(label);
}

function render() {
  renderList();
  renderCurrent();
  renderReadout();
}

function followPlayback() {
  renderReadout();
  if (currentLabel()?.task === 'hold') renderTimeline(currentLabel());
  if (!ui.video.paused) requestAnimationFrame(followPlayback);
}

// ---------------------------------------------------------------- files in and out

const readText = (file) => file.text();

async function loadToLabel(file) {
  const rows = parseCsv(await readText(file));
  if (!rows.length || !['video_id', 'label', 'task'].every((column) => column in rows[0])) {
    setStatus('Berkas ini bukan to_label.csv (butuh kolom video_id, label, task).', true);
    return;
  }
  state.allEntries = rows;
  state.index = -1;
  savePrefs();
  render();
  if (state.files.size) goNext(); // before setStatus: opening a video clears the status line
  setStatus(`${rows.length} video dimuat dari ${file.name}.`);
}

function loadFolder(fileList) {
  state.files = pickVideoFiles([...fileList]);
  const found = state.allEntries.filter((entry) => state.files.has(entry.video_id)).length;
  render();
  if (state.allEntries.length) goNext(); // before setStatus: opening a video clears the status line
  setStatus(`${found} dari ${state.allEntries.length} video di to_label.csv ditemukan di folder ini.`,
    found < state.allEntries.length);
}

async function importLabels(file) {
  const tasks = new Map(state.allEntries.map((entry) => [entry.video_id, entry.task]));
  const rows = parseCsv(await readText(file)).filter((row) => tasks.has(row.video_id));
  state.labels = { ...state.labels, ...Object.fromEntries(rows.map((row) => [row.video_id, fromRow(row, tasks.get(row.video_id))])) };
  saveSession();
  render();
  setStatus(`${rows.length} label diimpor dari ${file.name} ke sesi ${state.session}.`);
}

function exportCsv() {
  const labeler = ui.labeler.value.trim();
  if (!labeler) { setStatus('Isi nama pelabel dulu.', true); ui.labeler.focus(); return; }
  const rows = exportRows(entries(), state.labels, labeler);
  if (!rows.length) { setStatus('Belum ada label yang selesai untuk diekspor.', true); return; }
  const flawed = entries().filter((entry) => state.labels[entry.video_id]
    && labelProblems(state.labels[entry.video_id], Number(entry.duration_s) * 1000).length).length;

  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([toCsv(rows, REP_LABEL_COLUMNS)], { type: 'text/csv' }));
  link.download = EXPORT_NAMES[state.session];
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  setStatus(`${rows.length} label diekspor ke ${link.download}`
    + (flawed ? ` — ${flawed} masih bermasalah; validasi akan gagal sampai diperbaiki.` : '.'), flawed > 0);
}

function switchSession() {
  saveSession();
  state.session = ui.recheck.checked ? SESSION.RECHECK : SESSION.MAIN;
  ui.banner.hidden = state.session !== SESSION.RECHECK;
  loadSession();
  state.index = -1;
  render();
  if (state.files.size) goNext();
}

function seekFromTimeline(event) {
  const box = ui.timeline.getBoundingClientRect();
  if (!Number.isFinite(ui.video.duration)) return;
  ui.video.currentTime = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * ui.video.duration;
}

// ---------------------------------------------------------------- start

function init() {
  const prefs = readStore(PREFS_KEY) ?? {};
  ui.labeler.value = prefs.labeler ?? '';
  state.speed = prefs.speed ?? 1;
  state.allEntries = prefs.entries ?? [];
  loadSession();

  document.addEventListener('keydown', onKey);
  ui.toLabelFile.addEventListener('change', (e) => e.target.files[0] && loadToLabel(e.target.files[0]));
  ui.videoFolder.addEventListener('change', (e) => loadFolder(e.target.files));
  ui.importFile.addEventListener('change', (e) => e.target.files[0] && importLabels(e.target.files[0]));
  ui.labeler.addEventListener('input', savePrefs);
  ui.recheck.addEventListener('change', switchSession);
  ui.onlyTodo.addEventListener('change', renderList);
  ui.exportButton.addEventListener('click', exportCsv);
  ui.timeline.addEventListener('click', seekFromTimeline);
  ui.ambiguous.addEventListener('change', () => updateLabel((l) => setAmbiguous(l, ui.ambiguous.checked)));
  ui.notes.addEventListener('input', () => updateLabel((l) => setAmbiguous(l, l.ambiguous, ui.notes.value)));
  ui.video.addEventListener('loadedmetadata', () => { ui.video.playbackRate = state.speed; render(); });
  ui.video.addEventListener('timeupdate', renderReadout);
  ui.video.addEventListener('seeked', renderReadout);
  ui.video.addEventListener('play', () => requestAnimationFrame(followPlayback));
  ui.video.addEventListener('error', () => {
    if (!ui.video.getAttribute('src')) return;
    ui.videoError.hidden = false;
    ui.videoError.textContent = 'Browser ini tidak bisa memutar video ini — kemungkinan .MOV dengan codec HEVC (H.265). '
      + 'Coba Chrome/Edge terbaru, atau konversi: ffmpeg -i nama.MOV -c:v libx264 -crf 18 nama.mp4 '
      + '(nama dasar berkas jangan diubah). Bila tetap gagal, tandai ambigu.';
  });

  render();
  if (state.allEntries.length) setStatus(`${state.allEntries.length} video dari sesi sebelumnya. Pilih folder video untuk lanjut.`);
}

init();
