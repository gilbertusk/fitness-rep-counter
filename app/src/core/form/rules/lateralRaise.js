/** Lateral raise: the hands stop at shoulder height. Readable from the front as well. See docs/FORM_RULES.md. */
export default Object.freeze({
  view: 'any',
  sidePoints: ['shoulder', 'elbow', 'wrist', 'hip'],
  rules: Object.freeze([
    {
      id: 'lateral-raise-too-high',
      // How far the wrist rises above the shoulder, in torso lengths (0 = level with the shoulder).
      measure: { kind: 'heightAbove', points: ['wrist', 'shoulder'] },
      limit: { max: 0.15 },
      message: 'Jangan angkat tangan melewati bahu',
    },
  ]),
});
