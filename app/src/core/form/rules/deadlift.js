/** Deadlift: no rounding of the upper back, read through a 2D proxy. See docs/FORM_RULES.md for its limits. */
export default Object.freeze({
  view: 'side',
  sidePoints: ['ear', 'shoulder', 'hip', 'knee'],
  rules: Object.freeze([
    {
      id: 'deadlift-rounded-back',
      // Ear, shoulder and hip stay close to one line with a neutral spine; a rounding upper back
      // pushes the shoulder off that line and closes this angle.
      measure: { kind: 'angle', points: ['ear', 'shoulder', 'hip'] },
      limit: { min: 150 },
      message: 'Jaga punggung tetap lurus',
    },
  ]),
});
