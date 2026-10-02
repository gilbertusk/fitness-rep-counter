/** Renders the workout view (core/session/workout.js `view`) into the page. Display only. */

const PHASE_TEXT = Object.freeze({
  idle: 'Siap',
  detecting: 'Mengenali latihan…',
  counting: 'Menghitung',
  resting: 'Istirahat',
});

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
  if (view.confidence !== null) parts.push(`yakin ${Math.round(view.confidence * 100)}%`);
  if (view.manual) parts.push('dipilih manual');
  if (view.status === 'experimental') parts.push('eksperimental — belum dievaluasi');
  if (view.note) parts.push(view.note);
  return parts.join(' · ');
}

function formNote(view) {
  if (!view.label) return 'Pilih atau tunggu latihan dikenali untuk koreksi form.';
  if (!view.hasRules) return 'Belum ada aturan form untuk latihan ini — repetisi tetap dihitung.';
  if (view.warnings.length) return '';
  return view.formChecked ? 'Form terlihat baik.' : 'Form belum bisa dinilai — bagian tubuh yang diperlukan tidak terlihat.';
}

export function renderPanel(ui, view) {
  ui.phase.textContent = PHASE_TEXT[view.phase] ?? view.phase;
  ui.exerciseName.textContent = exerciseText(view);
  ui.exerciseMeta.textContent = exerciseMeta(view);

  const hold = view.task === 'hold';
  ui.repsCard.hidden = hold;
  ui.holdCard.hidden = !hold;
  ui.count.textContent = String(view.reps);
  ui.hold.textContent = formatDuration(view.hold.currentMs);
  ui.holdBest.textContent = view.hold.bestMs ? `terbaik ${formatDuration(view.hold.bestMs)}` : '';

  ui.warnings.replaceChildren(...view.warnings.map((text) => Object.assign(document.createElement('li'), { textContent: text })));
  ui.formNote.textContent = formNote(view);
  ui.status.textContent = view.quality ?? '';
  ui.fps.textContent = view.fps ? `${Math.round(view.fps)} FPS` : '';
}
