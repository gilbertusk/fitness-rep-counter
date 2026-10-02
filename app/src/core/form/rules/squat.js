/** Squat — ported unchanged from the original app (core/exercises.js, measure.js). See docs/FORM_RULES.md. */
export default Object.freeze({
  view: 'side',
  sidePoints: ['hip', 'knee', 'ankle'],
  rules: Object.freeze([
    {
      id: 'squat-torso-lean',
      measure: { kind: 'angleFromVertical', points: ['shoulder', 'hip'] },
      limit: { max: 45 },
      message: 'Punggung terlalu membungkuk',
    },
  ]),
});
