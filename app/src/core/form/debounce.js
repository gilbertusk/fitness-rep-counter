/**
 * A form warning appears only after it has held for DEBOUNCE_MS without a break, so one bad frame
 * of pose tracking never flashes a correction at the user. It disappears as soon as it stops holding.
 * Pure and immutable.
 */

export const DEBOUNCE_MS = 500;

export function createDebounce() {
  return Object.freeze({ since: Object.freeze({}) });
}

/**
 * @param {string[]} activeIds warnings whose rule is broken in this frame
 * @returns {{ state: object, visible: string[] }} visible keeps the order of activeIds
 */
export function updateDebounce(state, activeIds, timeMs, holdMs = DEBOUNCE_MS) {
  const since = Object.freeze(Object.fromEntries(activeIds.map((id) => [id, state.since[id] ?? timeMs])));
  return {
    state: Object.freeze({ since }),
    visible: activeIds.filter((id) => timeMs - since[id] >= holdMs),
  };
}
