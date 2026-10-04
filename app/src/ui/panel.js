/** Renders the workout view (core/session/workout.js `view`) into the page. Display only. */

const PHASE_TEXT = Object.freeze({
  idle: 'Siap',
  detecting: 'Mengenali latihan…',
  counting: 'Menghitung',
  resting: 'Istirahat',
});
const TOTAL_KEYPOINTS = 33;

export const formatDuration = (ms) => {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};

function exerciseText(view) {
  if (view.label) return view.name;
  if (view.phase === 'detecting' && view.autoAvailable) return 'Mengenali…';
  return view.autoAvailable ? '–' : 'Belum dipilih';
}

function exerciseMeta(view) {
  const parts = [];
  if (view.status === 'experimental') parts.push('eksperimental — belum dievaluasi');
  if (view.note) parts.push(view.note);
  return parts.join(' · ');
}

function modeText(view) {
  if (view.manual) return 'Dipilih manual';
  return view.autoAvailable ? 'Auto detect' : 'Pilih manual';
}

function formNote(view) {
  if (!view.label) return 'Pilih atau tunggu latihan dikenali untuk koreksi form.';
  if (!view.hasRules) return 'Belum ada aturan form untuk latihan ini — repetisi tetap dihitung.';
  if (view.warnings.length) return '';
  return view.formChecked ? 'Form terlihat baik.' : 'Form belum bisa dinilai — bagian tubuh yang diperlukan tidak terlihat.';
}

/** Restart a one-shot CSS animation (the ring sweep and number bump on each new rep). */
function replay(element, className) {
  element.classList.remove(className);
  void element.getBoundingClientRect();
  element.classList.add(className);
}

function renderReps(ui, view) {
  const previous = Number(ui.count.textContent);
  ui.count.textContent = String(view.reps);
  if (view.reps > previous) {
    replay(ui.repsRing, 'sweep');
    replay(ui.count, 'bump');
  }
  ui.repsNote.textContent = view.repsWithWarning ? `${view.repsWithWarning} rep dengan peringatan form` : '';
}

function renderConfidence(ui, view) {
  ui.confidenceRow.hidden = view.confidence === null;
  if (view.confidence === null) return;
  const percent = Math.round(view.confidence * 100);
  ui.confidenceText.textContent = `yakin ${percent}%`;
  ui.confidenceBar.style.width = `${percent}%`;
}

function renderTelemetry(ui, view) {
  ui.keypoints.textContent = view.visibleKeypoints ? `${view.visibleKeypoints} / ${TOTAL_KEYPOINTS}` : '–';
  ui.setTime.textContent = view.setDurationMs === null ? '–' : formatDuration(view.setDurationMs);
  ui.repsWarned.textContent = view.setDurationMs === null ? '–' : String(view.repsWithWarning);
}

function renderRest(ui, view) {
  ui.restBanner.hidden = !view.rest;
  if (!view.rest) return;
  const { name, reps, holdMs } = view.rest;
  const what = holdMs && !reps ? `tahan ${formatDuration(holdMs)}` : `${reps} rep`;
  ui.restSet.textContent = `${name ?? 'Latihan tidak dikenali'} — ${what}`;
  ui.restTime.textContent = formatDuration(view.rest.sinceMs);
}

export function renderPanel(ui, view) {
  ui.phase.textContent = PHASE_TEXT[view.phase] ?? view.phase;
  ui.stepper.dataset.phase = view.phase;
  ui.exerciseName.textContent = exerciseText(view);
  ui.exerciseMeta.textContent = exerciseMeta(view);
  ui.modeChip.textContent = modeText(view);
  renderConfidence(ui, view);

  const hold = view.task === 'hold';
  ui.repsCard.hidden = hold;
  ui.holdCard.hidden = !hold;
  renderReps(ui, view);
  ui.hold.textContent = formatDuration(view.hold.currentMs);
  ui.holdBest.textContent = view.hold.bestMs ? `terbaik ${formatDuration(view.hold.bestMs)}` : '';

  ui.warnings.replaceChildren(...view.warnings.map((text) => Object.assign(document.createElement('li'), { textContent: text })));
  const note = formNote(view);
  ui.formNote.textContent = note;
  ui.formNote.dataset.tone = view.formChecked && !view.warnings.length && view.hasRules && view.label ? 'good' : 'muted';

  renderTelemetry(ui, view);
  renderRest(ui, view);
  ui.status.textContent = view.quality ?? '';
  ui.fps.textContent = view.fps ? `${Math.round(view.fps)} FPS` : '';
}
