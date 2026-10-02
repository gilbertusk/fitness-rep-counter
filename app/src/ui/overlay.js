/** Creates a skeleton renderer for the overlay canvas. Draws whatever landmarks it is given. */
export function createPoseRenderer(canvas, vision) {
  const ctx = canvas.getContext('2d');
  const drawing = new vision.DrawingUtils(ctx);

  return function drawPose(landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks) return;

    drawing.drawConnectors(landmarks, vision.PoseLandmarker.POSE_CONNECTIONS, { color: '#60a5fa', lineWidth: 4 });
    drawing.drawLandmarks(landmarks, { color: '#f97316', radius: 4 });
  };
}
