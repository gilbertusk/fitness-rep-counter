/**
 * Rate limit for spoken feedback: at most one utterance every SPEECH_GAP_MS. Only the newest pending
 * message is kept — by the time the gap has passed, "tiga" is stale and "empat" is what matters.
 * The Web Speech API call itself lives in adapters/speech.js. Pure and immutable.
 */

export const SPEECH_GAP_MS = 2000;

export function createSpeechQueue({ enabled = true, minGapMs = SPEECH_GAP_MS } = {}) {
  return Object.freeze({ enabled, minGapMs, pending: null, lastSpokenMs: -Infinity });
}

/** Queue a message, replacing any message still waiting. Ignored while speech is off. */
export function say(state, text) {
  if (!state.enabled || !text) return state;
  return Object.freeze({ ...state, pending: String(text) });
}

/** The message to speak now, if the gap has passed. @returns {{ state: object, text: string|null }} */
export function nextUtterance(state, timeMs) {
  if (!state.enabled || state.pending === null || timeMs - state.lastSpokenMs < state.minGapMs) {
    return { state, text: null };
  }
  return { state: Object.freeze({ ...state, pending: null, lastSpokenMs: timeMs }), text: state.pending };
}

/** Turning speech off also drops whatever was waiting, so nothing stale is spoken when it comes back on. */
export function setSpeechEnabled(state, enabled) {
  return Object.freeze({ ...state, enabled: Boolean(enabled), pending: enabled ? state.pending : null });
}
