"""Unit tests for repcount.features.features (spec: docs/FEATURES.md)."""

import numpy as np
import pytest

from repcount.features import features as feat

STANDING = {
    0: (0.50, 0.10),
    11: (0.45, 0.25), 12: (0.55, 0.25),
    13: (0.45, 0.40), 14: (0.55, 0.40),
    15: (0.45, 0.55), 16: (0.55, 0.55),
    23: (0.46, 0.55), 24: (0.54, 0.55),
    25: (0.46, 0.75), 26: (0.54, 0.75),
    27: (0.46, 0.95), 28: (0.54, 0.95),
}


def pose_frame(points: dict, visibility: float = 1.0) -> np.ndarray:
    frame = np.full((1, 33, 4), np.nan)
    for index, (x, y) in points.items():
        frame[0, index] = (x, y, 0.0, visibility)
    return frame


def angles_of(points: dict, width: float = 640, height: float = 640) -> np.ndarray:
    normalized = feat.normalize_points(feat.select_landmarks(pose_frame(points), width, height))
    return feat.joint_angles(normalized)[0] * 180.0


# ---------------------------------------------------------------- resampling


def test_resample_halves_a_30fps_sequence():
    timestamps = np.round(np.arange(60) * 1000 / 30)
    indices = feat.resample_indices(timestamps)
    assert indices.tolist() == list(range(0, 60, 2))


def test_resample_keeps_every_frame_of_a_15fps_sequence():
    timestamps = np.round(np.arange(20) * 1000 / 15)
    assert feat.resample_indices(timestamps).tolist() == list(range(20))


def test_resample_breaks_ties_towards_the_lower_index():
    # Targets fall exactly between the two source frames at 0 ms and 133.33 ms.
    indices = feat.resample_indices(np.array([0.0, 1000 / 15, 2000 / 15]))
    assert indices.tolist() == [0, 1, 2]


def test_resample_handles_empty_and_single_frame_input():
    assert feat.resample_indices(np.array([])).tolist() == []
    assert feat.resample_indices(np.array([17.0])).tolist() == [0]


def test_resample_repeats_a_source_frame_when_the_video_stalls():
    # A 5 fps stretch cannot fill the 15 fps grid (7 targets over 400 ms), so frames are reused.
    indices = feat.resample_indices(np.array([0.0, 200.0, 400.0]))
    assert indices.tolist() == [0, 0, 1, 1, 1, 2, 2]


# ---------------------------------------------------------------- geometry


def test_select_landmarks_stretches_x_by_the_aspect_ratio_only():
    points = feat.select_landmarks(pose_frame(STANDING), 1280, 720)
    assert points.shape == (1, 13, 3)
    assert points[0, feat.NOSE, 0] == pytest.approx(0.50 * 1280 / 720)
    assert points[0, feat.NOSE, 1] == pytest.approx(0.10)


def test_select_landmarks_keeps_visibility_untouched():
    points = feat.select_landmarks(pose_frame(STANDING, visibility=0.31), 640, 640)
    assert points[0, :, 2] == pytest.approx(0.31)


def test_normalize_puts_the_hip_midpoint_at_the_origin():
    points = feat.normalize_points(feat.select_landmarks(pose_frame(STANDING), 640, 640))
    hip_mid = (points[0, feat.L_HIP, :2] + points[0, feat.R_HIP, :2]) / 2
    assert hip_mid == pytest.approx([0.0, 0.0], abs=1e-12)


def test_normalize_is_invariant_to_translation_and_zoom():
    moved = {i: (0.3 + 0.5 * x, 0.1 + 0.5 * y) for i, (x, y) in STANDING.items()}
    original = feat.normalize_points(feat.select_landmarks(pose_frame(STANDING), 640, 640))
    zoomed = feat.normalize_points(feat.select_landmarks(pose_frame(moved), 640, 640))
    assert zoomed[..., :2] == pytest.approx(original[..., :2], abs=1e-9)


def test_normalize_marks_a_frame_missing_when_the_torso_collapses():
    collapsed = {**STANDING, 11: STANDING[23], 12: STANDING[24]}
    points = feat.normalize_points(feat.select_landmarks(pose_frame(collapsed), 640, 640))
    assert np.isnan(points).all()


def test_normalize_survives_one_missing_side():
    one_side = {i: p for i, p in STANDING.items() if i not in (12, 24)}
    points = feat.normalize_points(feat.select_landmarks(pose_frame(one_side), 640, 640))
    assert np.isfinite(points[0, feat.L_HIP, :2]).all()


def test_straight_limbs_measure_180_degrees():
    angles = angles_of(STANDING)
    assert angles[0] == pytest.approx(180.0, abs=0.5)   # left elbow
    assert angles[6] == pytest.approx(180.0, abs=0.5)   # left knee


def test_a_right_angle_at_the_elbow_measures_90_degrees():
    bent = {**STANDING, 15: (0.60, 0.40), 16: (0.70, 0.40)}  # forearm horizontal, upper arm vertical
    assert angles_of(bent)[0] == pytest.approx(90.0, abs=0.5)


def test_angle_is_nan_when_a_point_is_missing_or_two_points_coincide():
    without_wrist = {i: p for i, p in STANDING.items() if i != 15}
    assert np.isnan(angles_of(without_wrist)[0])
    coincident = {**STANDING, 15: STANDING[13]}  # wrist exactly on the elbow
    assert np.isnan(angles_of(coincident)[0])


def test_aspect_correction_changes_a_diagonal_limb():
    diagonal = {**STANDING, 27: (0.30, 0.95), 28: (0.70, 0.95)}  # feet splayed → diagonal shins
    assert angles_of(diagonal, 640, 640)[6] != pytest.approx(angles_of(diagonal, 1280, 720)[6], abs=0.5)


def test_frame_features_uses_the_layout_of_the_spec():
    points = feat.normalize_points(feat.select_landmarks(pose_frame(STANDING), 640, 640))
    row = feat.frame_features(points)[0]
    assert row.shape == (feat.N_FEATURES,) == (47,)
    assert row[:39] == pytest.approx(points[0].reshape(39))
    assert row[39:] == pytest.approx(feat.joint_angles(points)[0])


# ---------------------------------------------------------------- gaps, windows, flip


def test_gaps_up_to_five_frames_are_filled_linearly():
    column = np.arange(12, dtype=float).reshape(12, 1)
    column[3:8] = np.nan  # a 5-frame gap between index 2 and index 8
    filled = feat.interpolate_gaps(column)
    assert filled[:, 0] == pytest.approx(np.arange(12, dtype=float))


def test_gaps_longer_than_five_frames_stay_missing():
    column = np.arange(12, dtype=float).reshape(12, 1)
    column[3:9] = np.nan  # 6 frames
    assert np.isnan(feat.interpolate_gaps(column)[3:9, 0]).all()


def test_leading_and_trailing_gaps_are_never_extrapolated():
    column = np.arange(10, dtype=float).reshape(10, 1)
    column[:2] = np.nan
    column[-2:] = np.nan
    filled = feat.interpolate_gaps(column)
    assert np.isnan(filled[:2, 0]).all() and np.isnan(filled[-2:, 0]).all()


def test_missing_frames_flags_any_remaining_nan():
    features = np.zeros((3, feat.N_FEATURES))
    features[1, 7] = np.nan
    assert feat.missing_frames(features).tolist() == [False, True, False]


def test_window_starts_follow_the_stride_and_need_a_full_window():
    assert feat.window_starts(48).tolist() == [0, 15]
    assert feat.window_starts(29).tolist() == []
    assert feat.window_starts(30).tolist() == [0]


def test_make_windows_replaces_leftover_nan_with_zero_and_reports_the_ratio():
    features = np.zeros((30, feat.N_FEATURES))
    features[:3] = np.nan
    cut = feat.make_windows(features)
    assert cut["windows"].shape == (1, 30, feat.N_FEATURES)
    assert not np.isnan(cut["windows"]).any()
    assert cut["missing_ratio"][0] == pytest.approx(0.1)


def test_a_window_is_usable_exactly_at_the_thirty_percent_boundary():
    assert feat.usable(np.array([0.30, 0.3001])).tolist() == [True, False]


def test_flip_swaps_sides_negates_x_and_is_its_own_inverse():
    points = feat.normalize_points(feat.select_landmarks(pose_frame(STANDING), 640, 640))
    row = feat.frame_features(points)
    flipped = feat.flip_features(row)
    left_x, right_x = feat.L_SHOULDER * 3, feat.R_SHOULDER * 3
    assert flipped[0, left_x] == pytest.approx(-row[0, right_x])
    assert flipped[0, 39] == pytest.approx(row[0, 40])  # left elbow angle ↔ right elbow angle
    assert feat.flip_features(flipped) == pytest.approx(row, nan_ok=True)


def test_flipping_a_mirrored_pose_reproduces_the_original():
    mirrored = {i: (1.0 - x, y) for i, (x, y) in STANDING.items()}
    swap = dict(zip(range(11, 29, 2), range(12, 30, 2), strict=False))
    relabelled = {swap.get(i, swap_back): p for i, p in mirrored.items()
                  for swap_back in [next((a for a, b in swap.items() if b == i), i)]}
    original = feat.frame_features(feat.normalize_points(feat.select_landmarks(pose_frame(STANDING), 640, 640)))
    other = feat.frame_features(feat.normalize_points(feat.select_landmarks(pose_frame(relabelled), 640, 640)))
    assert feat.flip_features(other)[..., :39] == pytest.approx(original[..., :39], abs=1e-9)


def test_sequence_features_runs_the_whole_pipeline():
    landmarks = np.repeat(pose_frame(STANDING), 60, axis=0)
    timestamps = np.round(np.arange(60) * 1000 / 30)
    features = feat.sequence_features(landmarks, timestamps, 1280, 720)
    assert features.shape == (30, feat.N_FEATURES)


def test_sequence_features_returns_nothing_for_an_empty_video():
    assert feat.sequence_features(np.empty((0, 33, 4)), np.array([]), 640, 480).shape == (0, feat.N_FEATURES)
