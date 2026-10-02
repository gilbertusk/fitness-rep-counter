/** Push-up — ported unchanged from the original app (core/exercises.js, measure.js). See docs/FORM_RULES.md. */
export default Object.freeze({
  view: 'side',
  sidePoints: ['shoulder', 'elbow', 'wrist'],
  rules: Object.freeze([
    {
      id: 'push-up-body-line',
      measure: { kind: 'angle', points: ['shoulder', 'hip', 'ankle'] },
      limit: { min: 160 },
      message: 'Jaga badan tetap lurus',
    },
  ]),
});
