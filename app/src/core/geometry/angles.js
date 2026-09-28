const RAD_TO_DEG = 180 / Math.PI;

/**
 * Angle at point b (in degrees) formed by segments b→a and b→c.
 * Returns null when a point is missing or two points coincide.
 */
export function calculateAngle(a, b, c) {
  if (!a || !b || !c) return null;

  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const magnitude = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (magnitude === 0) return null;

  const cosTheta = (ab.x * cb.x + ab.y * cb.y) / magnitude;
  return Math.acos(Math.max(-1, Math.min(1, cosTheta))) * RAD_TO_DEG;
}

/** Angle (in degrees) between segment top→bottom and the vertical axis. */
export function angleFromVertical(top, bottom) {
  const dx = Math.abs(bottom.x - top.x);
  const dy = Math.abs(bottom.y - top.y);
  return Math.atan2(dx, dy) * RAD_TO_DEG;
}
