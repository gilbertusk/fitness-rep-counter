"""Unit tests for repcount.export.keypoints_json."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from repcount.export import keypoints_json as kj
from repcount.features.golden import synthetic_sequence


def write_npz(root, label: str, video_id: str) -> None:
    sequence = synthetic_sequence()
    path = root / label / f"{video_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, landmarks=sequence["landmarks"].astype(np.float32),
                        timestamps_ms=sequence["timestamps_ms"], width=np.int32(1280), height=np.int32(720),
                        label=np.str_(label), video_id=np.str_(video_id))


def test_a_detected_frame_becomes_rounded_nested_lists():
    frame = np.full((33, 4), 0.123456789)
    out = kj.frame_to_json(frame)
    assert len(out) == 33 and len(out[0]) == 4
    assert out[0][0] == 0.12346


def test_a_frame_without_a_pose_becomes_null():
    assert kj.frame_to_json(np.full((33, 4), np.nan)) is None


def test_one_missing_landmark_keeps_the_frame_with_a_null_in_place():
    frame = np.full((33, 4), 0.5)
    frame[17] = np.nan  # e.g. a hand landmark the pipeline never fills
    out = kj.frame_to_json(frame)
    assert out is not None
    assert out[17] == [None, None, None, None]
    assert out[0] == [0.5, 0.5, 0.5, 0.5]


def test_a_missing_z_or_visibility_alone_does_not_drop_the_frame():
    frame = np.full((33, 4), 0.5)
    frame[:, 2] = np.nan
    out = kj.frame_to_json(frame)
    assert out is not None and out[0][2] is None


def test_a_video_keeps_its_size_timestamps_and_every_frame():
    sequence = {**synthetic_sequence(), "label": "squat", "video_id": "squat_1"}
    video = kj.video_to_json(sequence)
    assert (video["width"], video["height"]) == (1280, 720)
    assert len(video["landmarks"]) == len(video["timestamps_ms"]) == len(sequence["timestamps_ms"])
    gaps = [i for i, frame in enumerate(video["landmarks"]) if frame is None]
    assert gaps, "the synthetic sequence has NaN gaps, which must survive as null"
    assert len(gaps) < len(video["landmarks"]) // 2, "frames with a pose must not be dropped"
    assert json.loads(json.dumps(video)) == video, "strict JSON: no NaN literals"


def test_only_labelled_videos_with_keypoints_are_exported(tmp_path):
    write_npz(tmp_path / "kp", "squat", "squat_1")
    written, missing = kj.export(["squat_1", "squat_9"], tmp_path / "kp", tmp_path / "out")
    assert [p.name for p in written] == ["squat_1.json"]
    assert missing == ["squat_9"]


def test_the_cli_reads_the_labels_and_reports_what_is_missing(tmp_path, monkeypatch, capsys):
    write_npz(tmp_path / "kp", "squat", "squat_1")
    pd.DataFrame({"video_id": ["squat_1", "squat_2"], "label": ["squat", "squat"]}).to_csv(
        tmp_path / "labels.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--labels", str(tmp_path / "labels.csv"),
                                      "--keypoints-dir", str(tmp_path / "kp"), "--out-dir", str(tmp_path / "out")])
    kj.main()
    printed = capsys.readouterr().out
    assert "1 labelled videos" in printed and "no keypoints for squat_2" in printed
    assert json.loads((tmp_path / "out" / "squat_1.json").read_text())["label"] == "squat"


def test_the_cli_explains_missing_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--labels", str(tmp_path / "nope.csv")])
    with pytest.raises(SystemExit, match="label the videos first"):
        kj.main()
