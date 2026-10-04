import { exerciseName } from '../core/exercises.js';
import { formatDuration } from './panel.js';

/** Renders finished sets, newest first. Display only; storage lives in adapters/storage.js. */

const element = (tag, className, text) => Object.assign(document.createElement(tag), { className, textContent: text ?? '' });

function detail(set) {
  const parts = set.holdMs && !set.reps ? [`tahan ${formatDuration(set.holdMs)}`] : [`${set.reps} rep`, formatDuration(set.durationMs)];
  if (set.savedAt) parts.push(new Date(set.savedAt).toLocaleString('id-ID', { dateStyle: 'short', timeStyle: 'short' }));
  return parts.join(' · ');
}

/** What the set's own records say about form: reps without a warning. No score is invented. */
function badge(set) {
  if (!set.reps) return null;
  const clean = set.reps - (set.repsWithWarning ?? 0);
  return element('span', `set-badge ${clean === set.reps ? 'clean' : 'warned'}`, `${clean}/${set.reps} rep tanpa peringatan`);
}

function row(set, number, fresh) {
  const item = element('li', fresh ? 'fresh' : '');
  const main = element('div', 'set-main');
  main.append(element('span', 'set-name', set.label ? exerciseName(set.label) : 'Latihan tidak dikenali'),
    element('span', 'set-detail', detail(set)));
  item.append(element('span', 'set-index', `#${number}`), main);
  const tag = badge(set);
  if (tag) item.append(tag);
  return item;
}

/**
 * @param {HTMLElement} [count] optional element for "N set"
 * @param {boolean} [highlightNewest] flash the newest row (a set that just finished)
 */
export function renderHistory(list, empty, sets, count = null, highlightNewest = false) {
  list.replaceChildren(...sets.map((set, i) => row(set, i + 1, highlightNewest && i === sets.length - 1)).reverse());
  empty.hidden = sets.length > 0;
  if (count) count.textContent = sets.length ? `${sets.length} set` : '';
}
