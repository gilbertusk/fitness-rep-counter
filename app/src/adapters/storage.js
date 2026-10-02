/**
 * Set history in localStorage. Storage can be missing or refuse writes (private windows, blocked site
 * data, a full quota), so every access is wrapped: the app keeps working, it just forgets on reload.
 */

const KEY = 'repCounter:v1:history';
const LIMIT = 200; // keep the newest sets; one set is ~150 bytes

export function loadHistory() {
  try {
    const sets = JSON.parse(localStorage.getItem(KEY) ?? '[]');
    return Array.isArray(sets) ? sets : [];
  } catch {
    return [];
  }
}

/** @returns {boolean} false when the browser refused to store it */
export function saveHistory(sets) {
  try {
    localStorage.setItem(KEY, JSON.stringify(sets.slice(-LIMIT)));
    return true;
  } catch {
    return false;
  }
}

export function clearStoredHistory() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // nothing stored, nothing to clear
  }
}
