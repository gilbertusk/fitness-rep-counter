/** Shoulder press: full lockout at the top of every rep. Judged once per rep. See docs/FORM_RULES.md. */
export default Object.freeze({
  view: 'any',
  sidePoints: ['shoulder', 'elbow', 'wrist'],
  rules: Object.freeze([
    {
      id: 'shoulder-press-lockout',
      measure: { kind: 'angle', points: ['shoulder', 'elbow', 'wrist'] },
      // Evaluated at the end of each rep on the largest elbow angle seen during it.
      per: 'rep',
      limit: { min: 160 },
      message: 'Luruskan lengan penuh di atas kepala',
    },
  ]),
});
