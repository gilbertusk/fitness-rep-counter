"""Window features for the exercise classifier — implementation of `docs/FEATURES.md`.

The same specification is implemented in `app/src/core/features/features.js`; both are pinned to
`app/tests/fixtures/features_golden.json` by a parity test on each side. Change one, change all three.

Pure functions only: no file I/O, no global state.
"""

import numpy as np

# MediaPipe indices of the 13 landmarks we keep, in the fixed order of docs/FEATURES.md §4.
LANDMARK_INDICES = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = range(7)
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = range(7, 13)

# Angle at b between b→a and b→c, in the fixed order of docs/FEATURES.md §5.
ANGLE_TRIPLETS = (
    (L_SHOULDER, L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW, R_WRIST),
    (L_ELBOW, L_SHOULDER, L_HIP),
    (R_ELBOW, R_SHOULDER, R_HIP),
    (L_SHOULDER, L_HIP, L_KNEE),
    (R_SHOULDER, R_HIP, R_KNEE),
    (L_HIP, L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE, R_ANKLE),
)

# Left↔right swaps for the flip augmentation (docs/FEATURES.md §9); nose maps to itself.
FLIP_POINT_PAIRS = ((L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
                    (L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE))
FLIP_ANGLE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))

N_POINTS = len(LANDMARK_INDICES)
N_ANGLES = len(ANGLE_TRIPLETS)
N_FEATURES = N_POINTS * 3 + N_ANGLES  # 13 × (x, y, visibility) + 8 angles = 47

TARGET_FPS = 15
WINDOW_FRAMES = 30
WINDOW_STRIDE = 15
MAX_INTERPOLATION_GAP = 5
MAX_MISSING_RATIO = 0.30
MIN_SCALE = 1e-6


# ---------------------------------------------------------------- resampling


def resample_indices(timestamps_ms, target_fps: float = TARGET_FPS) -> np.ndarray:
    """Source frame index nearest to each 1/target_fps grid point (§2). Ties pick the lower index."""
    t = np.asarray(timestamps_ms, dtype=np.float64)
    if t.size == 0:
        return np.empty(0, dtype=np.int64)
    step = 1000.0 / target_fps
    n_targets = int(np.floor((t[-1] - t[0]) / step)) + 1
    targets = t[0] + np.arange(n_targets) * step
    after = np.searchsorted(t, targets)
    lower = np.clip(after - 1, 0, t.size - 1)
    upper = np.clip(after, 0, t.size - 1)
    take_lower = np.abs(t[lower] - targets) <= np.abs(t[upper] - targets)
    return np.where(take_lower, lower, upper).astype(np.int64)


# ---------------------------------------------------------------- per-frame geometry


def select_landmarks(landmarks: np.ndarray, width: float, height: float) -> np.ndarray:
    """(T, 33, 4) MediaPipe → (T, 13, 3) of aspect-corrected x, y and visibility (§1, §4)."""
    chosen = np.asarray(landmarks, dtype=np.float64)[:, list(LANDMARK_INDICES), :]
    points = chosen[..., [0, 1, 3]].copy()
    points[..., 0] *= float(width) / float(height)
    return points


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmean without the all-NaN warning: an all-NaN slice legitimately means 'missing'."""
    with np.errstate(invalid="ignore"):
        counts = np.sum(np.isfinite(values), axis=axis)
        totals = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=axis)
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center on the hip midpoint, scale by the mean shoulder–hip distance (§3).

    Frames whose center or scale cannot be computed become all-NaN.
    """
    xy = points[..., :2]
    hip_mid = _nanmean(np.stack([xy[:, L_HIP], xy[:, R_HIP]], axis=1), axis=1)
    torso = np.stack([
        np.linalg.norm(xy[:, L_SHOULDER] - xy[:, L_HIP], axis=-1),
        np.linalg.norm(xy[:, R_SHOULDER] - xy[:, R_HIP], axis=-1),
    ], axis=1)
    scale = _nanmean(torso, axis=1)

    usable = np.isfinite(scale) & (scale >= MIN_SCALE) & np.isfinite(hip_mid).all(axis=1)
    out = points.copy()
    with np.errstate(invalid="ignore"):
        out[..., :2] = (xy - hip_mid[:, None, :]) / np.where(usable, scale, np.nan)[:, None, None]
    out[~usable] = np.nan
    return out


def joint_angles(points: np.ndarray) -> np.ndarray:
    """(T, 13, >=2) → (T, 8) joint angles in [0, 1] (degrees / 180); NaN when undefined (§5)."""
    xy = points[..., :2]
    angles = np.full((points.shape[0], N_ANGLES), np.nan)
    for i, (a, b, c) in enumerate(ANGLE_TRIPLETS):
        ba, bc = xy[:, a] - xy[:, b], xy[:, c] - xy[:, b]
        magnitude = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.sum(ba * bc, axis=-1) / np.where(magnitude > 0, magnitude, np.nan)
            angles[:, i] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) / 180.0
    return angles


def frame_features(points: np.ndarray) -> np.ndarray:
    """(T, 13, 3) normalized points → (T, 47) feature rows in the fixed layout of §6."""
    flat = points.reshape(points.shape[0], N_POINTS * 3)
    return np.concatenate([flat, joint_angles(points)], axis=1)


# ---------------------------------------------------------------- gaps and windows


def interpolate_gaps(features: np.ndarray, max_gap: int = MAX_INTERPOLATION_GAP) -> np.ndarray:
    """Linearly fill NaN runs of at most `max_gap` frames, per column, never extrapolating (§7)."""
    out = np.array(features, dtype=np.float64, copy=True)
    for column in range(out.shape[1]):
        series = out[:, column]
        valid = np.flatnonzero(np.isfinite(series))
        for start, end in zip(valid[:-1], valid[1:], strict=False):
            gap = end - start - 1
            if 0 < gap <= max_gap:
                positions = np.arange(start + 1, end)
                series[start + 1:end] = np.interp(positions, [start, end], [series[start], series[end]])
    return out


def missing_frames(features: np.ndarray) -> np.ndarray:
    """(T, 47) → (T,) boolean: a frame is missing when any feature is still NaN (§7)."""
    return ~np.isfinite(features).all(axis=1)


def window_starts(n_frames: int, length: int = WINDOW_FRAMES, stride: int = WINDOW_STRIDE) -> np.ndarray:
    if n_frames < length:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_frames - length + 1, stride, dtype=np.int64)


def make_windows(features: np.ndarray, length: int = WINDOW_FRAMES, stride: int = WINDOW_STRIDE) -> dict:
    """Cut a (T, 47) sequence into windows (§8).

    Returns `windows` (N, length, 47) with remaining NaN replaced by 0, the `starts` of each window,
    and `missing_ratio`, the fraction of missing frames used to drop or flag a window.
    """
    starts = window_starts(features.shape[0], length, stride)
    missing = missing_frames(features)
    windows = np.stack([features[s:s + length] for s in starts]) if starts.size else np.empty((0, length, N_FEATURES))
    ratios = np.array([missing[s:s + length].mean() for s in starts])
    return {"windows": np.nan_to_num(windows, nan=0.0), "starts": starts, "missing_ratio": ratios}


def usable(missing_ratio: np.ndarray, max_ratio: float = MAX_MISSING_RATIO) -> np.ndarray:
    """Windows kept for training and evaluation; the rest are only flagged at inference (§8)."""
    return np.asarray(missing_ratio) <= max_ratio


# ---------------------------------------------------------------- pipeline and augmentation


def sequence_features(landmarks: np.ndarray, timestamps_ms: np.ndarray, width: float, height: float) -> np.ndarray:
    """Raw MediaPipe sequence → (K, 47) features at TARGET_FPS, gaps filled (§1–§7)."""
    indices = resample_indices(timestamps_ms)
    if indices.size == 0:
        return np.empty((0, N_FEATURES))
    points = normalize_points(select_landmarks(np.asarray(landmarks)[indices], width, height))
    return interpolate_gaps(frame_features(points))


def flip_features(features: np.ndarray) -> np.ndarray:
    """Mirror left↔right: swap paired landmarks and angles, negate normalized x (§9)."""
    out = np.array(features, dtype=np.float64, copy=True)
    for left, right in FLIP_POINT_PAIRS:
        for offset in range(3):
            a, b = left * 3 + offset, right * 3 + offset
            out[..., [a, b]] = out[..., [b, a]]
    for left, right in FLIP_ANGLE_PAIRS:
        a, b = N_POINTS * 3 + left, N_POINTS * 3 + right
        out[..., [a, b]] = out[..., [b, a]]
    out[..., 0:N_POINTS * 3:3] *= -1.0
    return out
