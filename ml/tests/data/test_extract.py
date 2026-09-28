from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from repcount import config
from repcount.data import extract
from repcount.data.extract import (
    keep_frame,
    output_path,
    pose_to_arrays,
    resize_dims,
    sample_frame_indices,
    select_pending,
    timestamp_ms,
    write_errors,
)


def test_60fps_is_halved_to_30():
    indices = sample_frame_indices(120, 60.0, 30)
    assert indices == list(range(0, 120, 2))


def test_24fps_keeps_every_frame():
    assert sample_frame_indices(48, 24.0, 30) == list(range(48))


@pytest.mark.parametrize("fps", [29.97, 30.0, 30.05])
def test_ntsc_and_near_30fps_keep_every_frame(fps):
    assert sample_frame_indices(300, fps, 30) == list(range(300))


def test_50fps_sampled_to_30fps_keeps_the_right_rate():
    indices = sample_frame_indices(500, 50.0, 30)  # 10 s
    assert len(indices) == 300
    assert np.all(np.diff(indices) >= 1) and np.all(np.diff(indices) <= 2)


def test_keep_frame_always_keeps_the_first_frame():
    assert keep_frame(0, 120.0, 30)


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1920, 1080), (640, 360)), ((1080, 1920), (360, 640)), ((320, 240), (320, 240)), ((640, 480), (640, 480))],
)
def test_resize_dims_caps_longest_side_without_upscaling(size, expected):
    assert resize_dims(*size, 640) == expected


def test_timestamps_follow_source_frame_index():
    assert [timestamp_ms(i, 29.97) for i in (0, 1, 30)] == [0, 33, 1001]


def test_pose_to_arrays_is_nan_without_pose():
    landmarks, world = pose_to_arrays(SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]))
    assert landmarks.shape == (33, 4) and world.shape == (33, 3)
    assert landmarks.dtype == np.float32 and np.isnan(landmarks).all() and np.isnan(world).all()


def test_pose_to_arrays_copies_coordinates_and_visibility():
    point = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9)
    result = SimpleNamespace(pose_landmarks=[[point] * 33], pose_world_landmarks=[[point] * 33])
    landmarks, world = pose_to_arrays(result)
    np.testing.assert_allclose(landmarks[5], [0.1, 0.2, 0.3, 0.9], rtol=1e-6)
    np.testing.assert_allclose(world[5], [0.1, 0.2, 0.3], rtol=1e-6)


def test_select_pending_skips_existing_outputs_unless_overwrite(tmp_path):
    manifest = pd.DataFrame({"video_id": ["a_1", "a_2", "b_1"], "label": ["a", "a", "b"], "path": ["x", "y", "z"]})
    done = output_path(tmp_path, "a", "a_1")
    done.parent.mkdir(parents=True)
    done.write_bytes(b"")

    assert [r["video_id"] for r in select_pending(manifest, tmp_path, False, None)] == ["a_2", "b_1"]
    assert [r["video_id"] for r in select_pending(manifest, tmp_path, True, 2)] == ["a_1", "a_2"]


def test_write_errors_replaces_rows_of_retried_videos(tmp_path):
    path = tmp_path / "errors.csv"
    base = {"label": "a", "path": "p"}
    write_errors(
        [
            {**base, "video_id": "a_1", "ok": False, "error": "boom"},
            {**base, "video_id": "a_2", "ok": False, "error": "bad"},
        ],
        path,
    )

    n_failed = write_errors([{**base, "video_id": "a_1", "ok": True, "error": ""}], path)

    assert n_failed == 0
    assert pd.read_csv(path)["video_id"].tolist() == ["a_2"]


def test_process_one_reports_errors_instead_of_raising(tmp_path):
    row = {"video_id": "a_1", "label": "a", "path": "missing.mp4"}
    result = extract.process_one(row, str(tmp_path), str(tmp_path / "kp"), str(tmp_path / "model.task"))
    assert result["ok"] is False and "cannot open video" in result["error"]


@pytest.mark.skipif(not config.POSE_MODEL_PATH.exists(), reason="pose model not downloaded")
def test_extract_writes_npz_with_expected_schema(tmp_path, tiny_video):
    tiny_video(tmp_path / "videos" / "a" / "a_1.avi", n_frames=12, fps=60.0, size=(96, 64))
    row = {"video_id": "a_1", "label": "a", "path": "a/a_1.avi"}

    result = extract.process_one(row, str(tmp_path / "videos"), str(tmp_path / "kp"), str(config.POSE_MODEL_PATH))

    assert result["ok"], result.get("traceback")
    with np.load(output_path(tmp_path / "kp", "a", "a_1")) as data:
        assert data["landmarks"].shape == (6, 33, 4) and data["landmarks"].dtype == np.float32
        assert data["world_landmarks"].shape == (6, 33, 3)
        assert data["timestamps_ms"].tolist() == [0, 33, 67, 100, 133, 167]
        assert float(data["fps_effective"]) == pytest.approx(30.0)
        assert (int(data["width"]), int(data["height"])) == (96, 64)
        assert str(data["label"]) == "a" and str(data["video_id"]) == "a_1"
