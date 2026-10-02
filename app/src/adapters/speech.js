/**
 * Web Speech API wrapper. No timing logic here: core/session/speechQueue.js decides when to speak.
 * @returns {{ supported: boolean, speak: (text: string) => void }}
 */
export function createSpeaker(lang = 'id-ID') {
  const synth = globalThis.speechSynthesis;
  const Utterance = globalThis.SpeechSynthesisUtterance;
  if (!synth || !Utterance) return { supported: false, speak: () => {} };

  return {
    supported: true,
    speak(text) {
      synth.cancel(); // a newer message replaces one still being read out
      const utterance = new Utterance(text);
      utterance.lang = lang;
      synth.speak(utterance);
    },
  };
}
