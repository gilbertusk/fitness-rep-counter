/** Barbell biceps curl: the upper arm stays by the torso. See docs/FORM_RULES.md. */
export default Object.freeze({
  view: 'side',
  sidePoints: ['shoulder', 'elbow', 'wrist', 'hip'],
  rules: Object.freeze([
    {
      id: 'curl-elbow-forward',
      // Angle at the shoulder between the upper arm and the torso: grows as the elbow swings forward.
      measure: { kind: 'angle', points: ['elbow', 'shoulder', 'hip'] },
      limit: { max: 30 },
      message: 'Siku jangan maju — jaga lengan atas di samping badan',
    },
  ]),
});
