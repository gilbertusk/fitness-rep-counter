"""Unit tests for repcount.features.windows."""

import json

import numpy as np
import pytest

from repcount.features import features as feat
from repcount.features import windows
from repcount.features.golden import synthetic_sequence


@pytest.fixture
def keypoints_dir(tmp_path):
    """Factory writing synthetic .npz files in the layout repcount.data.extract produces."""

    def make(videos: dict[str, str]) -> "object":
        sequence = synthetic_sequence()
        for video_id, label in videos.items():
            path = tmp_path / label / f"{video_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                landmarks=sequence["landmarks"].astype(np.float32),
                timestamps_ms=sequence["timestamps_ms"],
                width=np.int32(sequence["width"]),
                height=np.int32(sequence["height"]),
                label=np.str_(label),
                video_id=np.str_(video_id),
            )
        return tmp_path

    return make


@pytest.fixture
def split_file(tmp_path):
    def make(splits: dict) -> "object":
        path = tmp_path / "split_test.json"
        path.write_text(json.dumps({"version": "test", "splits": splits}), encoding="utf-8")
        return path

    return make


def test_find_keypoint_files_indexes_every_class_folder(keypoints_dir):
    root = keypoints_dir({"a_1": "squat", "b_2": "push_up"})
    assert sorted(windows.find_keypoint_files(root)) == ["a_1", "b_2"]


def test_video_windows_drops_the_windows_with_too_many_missing_frames(keypoints_dir):
    root = keypoints_dir({"a_1": "squat"})
    result = windows.video_windows(windows.load_sequence(root / "squat" / "a_1.npz"))
    # The synthetic sequence is built to yield 2 windows of which the second is > 30% missing.
    assert result["n_frames"] == 48
    assert len(result["windows"]) == 1
    assert result["n_dropped"] == 1
    assert result["starts"].tolist() == [0]


def test_stack_dataset_labels_every_window_with_its_video():
    per_video = [
        {"windows": np.zeros((2, feat.WINDOW_FRAMES, feat.N_FEATURES)), "label": "squat", "video_id": "a_1"},
        {"windows": np.zeros((3, feat.WINDOW_FRAMES, feat.N_FEATURES)), "label": "push_up", "video_id": "b_2"},
    ]
    data = windows.stack_dataset(per_video, {"squat": 0, "push_up": 1})
    assert data["x"].shape == (5, feat.WINDOW_FRAMES, feat.N_FEATURES)
    assert data["y"].tolist() == [0, 0, 1, 1, 1]
    assert data["video_ids"].tolist() == ["a_1", "a_1", "b_2", "b_2", "b_2"]


def test_stack_dataset_handles_an_empty_split():
    data = windows.stack_dataset([], {"squat": 0})
    assert data["x"].shape == (0, feat.WINDOW_FRAMES, feat.N_FEATURES)
    assert data["y"].tolist() == []


def test_build_split_only_reads_the_videos_of_that_split(keypoints_dir, split_file):
    root = keypoints_dir({"a_1": "squat", "b_2": "push_up", "c_3": "squat"})
    document = json.loads(split_file({"train": ["a_1", "c_3"], "val": ["b_2"], "test": []}).read_text())
    train = windows.build_split(document, "train", root, {"squat": 0, "push_up": 1})
    assert train["n_videos"] == 2
    assert sorted(set(train["video_ids"].tolist())) == ["a_1", "c_3"]
    assert train["y"].tolist() == [0, 0]


def test_build_split_ignores_ids_without_keypoints(keypoints_dir, split_file):
    root = keypoints_dir({"a_1": "squat"})
    document = json.loads(split_file({"train": ["a_1", "missing_9"], "val": [], "test": []}).read_text())
    assert windows.build_split(document, "train", root, {"squat": 0})["n_videos"] == 1


def test_load_all_returns_every_split_and_a_sorted_vocabulary(keypoints_dir, split_file):
    root = keypoints_dir({"a_1": "squat", "b_2": "push_up"})
    path = split_file({"train": ["b_2"], "val": ["a_1"], "test": []})
    data = windows.load_all(path, root)
    assert data["labels"] == ["push_up", "squat"]
    assert data["train"]["n_videos"] == 1 and data["val"]["n_videos"] == 1
    assert data["test"]["x"].shape[0] == 0


def test_load_all_respects_the_limit(keypoints_dir, split_file):
    root = keypoints_dir({"a_1": "squat", "c_3": "squat"})
    path = split_file({"train": ["a_1", "c_3"], "val": [], "test": []})
    assert windows.load_all(path, root, limit=1)["train"]["n_videos"] == 1
