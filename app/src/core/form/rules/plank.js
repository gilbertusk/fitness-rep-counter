/** Plank: shoulder, hip and ankle in one line. Also decides "in position" for the hold timer. See docs/FORM_RULES.md. */
export default Object.freeze({
  view: 'side',
  sidePoints: ['shoulder', 'hip', 'ankle'],
  rules: Object.freeze([
    {
      id: 'plank-hips-low',
      // Hip's height relative to the shoulder–ankle line, in line lengths; positive = below the line.
      measure: { kind: 'lineOffset', points: ['shoulder', 'hip', 'ankle'] },
      limit: { max: 0.1 },
      message: 'Pinggul turun — kencangkan perut',
    },
    {
      id: 'plank-hips-high',
      measure: { kind: 'lineOffset', points: ['shoulder', 'hip', 'ankle'] },
      limit: { min: -0.12 },
      message: 'Pinggul terlalu tinggi',
    },
  ]),
});
