import { FEEDBACK } from '../core/counting/thresholdCounter.js';

/** Creates a skeleton renderer for the overlay canvas. */
export function createPoseRenderer(canvas, vision) {
  const ctx = canvas.getContext('2d');
  const drawing = new vision.DrawingUtils(ctx);

  return function drawPose(pose) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!pose) return;

    drawing.drawConnectors(pose, vision.PoseLandmarker.POSE_CONNECTIONS, { color: '#60a5fa', lineWidth: 4 });
    drawing.drawLandmarks(pose, { color: '#f97316', radius: 4 });
  };
}

/** Renders the rep count, joint angle, feedback and form warnings. */
export function renderStats(ui, counter, { angle, warnings }) {
  ui.count.textContent = counter.count;
  ui.angle.textContent = angle === null ? '–' : `${Math.round(angle)}°`;
  ui.feedback.textContent = counter.feedback ?? '–';
  ui.feedback.className = `feedback ${counter.feedback === FEEDBACK.GOOD ? 'good' : counter.feedback ? 'bad' : ''}`;
  ui.warnings.replaceChildren(...warnings.map((text) => Object.assign(document.createElement('li'), { textContent: text })));
}
