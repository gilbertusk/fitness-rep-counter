import { exerciseName } from '../core/exercises.js';
import { formatDuration } from './panel.js';

/** Renders finished sets, newest first. Display only; storage lives in adapters/storage.js. */

function describe(set) {
  const name = set.label ? exerciseName(set.label) : 'Latihan tidak dikenali';
  if (set.holdMs && !set.reps) return `${name} — tahan ${formatDuration(set.holdMs)}`;
  const warned = set.repsWithWarning ? ` (${set.repsWithWarning} dengan peringatan)` : '';
  return `${name} — ${set.reps} rep${warned} · ${formatDuration(set.durationMs)}`;
}

export function renderHistory(list, empty, sets) {
  list.replaceChildren(...[...sets].reverse().map((set) => {
    const item = document.createElement('li');
    item.textContent = describe(set);
    if (set.savedAt) {
      const when = Object.assign(document.createElement('span'), { className: 'when' });
      when.textContent = new Date(set.savedAt).toLocaleString('id-ID', { dateStyle: 'short', timeStyle: 'short' });
      item.append(when);
    }
    return item;
  }));
  empty.hidden = sets.length > 0;
}
