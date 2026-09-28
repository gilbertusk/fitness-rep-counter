from pathlib import Path

import pandas as pd
import pytest

from repcount.data.manifest import (
    MANIFEST_COLUMNS,
    build_manifest,
    ensure_unique_ids,
    list_videos,
    make_video_id,
    normalize_label,
    probe_video,
)


@pytest.mark.parametrize(
    ("folder", "label"),
    [
        ("pull Up", "pull_up"),
        ("push-up", "push_up"),
        ("tricep Pushdown", "tricep_pushdown"),
        ("barbell biceps curl", "barbell_biceps_curl"),
        ("  T Bar  Row ", "t_bar_row"),
    ],
)
def test_normalize_label_to_snake_case(folder, label):
    assert normalize_label(folder) == label


def test_make_video_id_uses_the_snake_case_stem():
    assert make_video_id(Path("push-up/push-up_17.mp4")) == "push_up_17"
    assert make_video_id(Path("x/Barbell_Hip_Thrust__glute.MOV")) == "barbell_hip_thrust_glute"


def test_list_videos_reads_class_folders_and_skips_other_files(tmp_path):
    for rel in ["pull Up/pull up_1.mp4", "pull Up/pull up_2.MOV", "pull Up/notes.txt", "squat/squat_1.mov"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")

    records = list_videos(tmp_path)

    assert [r["video_id"] for r in records] == ["pull_up_1", "pull_up_2", "squat_1"]
    assert records[1] == {"video_id": "pull_up_2", "label": "pull_up", "path": "pull Up/pull up_2.MOV", "ext": "mov"}


def test_duplicate_video_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate video_id"):
        ensure_unique_ids([{"video_id": "a_1", "path": "a/a_1.mp4"}, {"video_id": "a_1", "path": "a/a-1.mov"}])


def test_unreadable_video_is_recorded_not_raised(tmp_path):
    broken = tmp_path / "squat" / "squat_1.mp4"
    broken.parent.mkdir()
    broken.write_bytes(b"not a video")

    info = probe_video(broken)
    manifest = build_manifest(tmp_path)

    assert info["error"] and info["size_bytes"] == 11
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert manifest.loc[0, "error"] != ""


def test_probe_reads_metadata_of_a_real_video(tmp_path, tiny_video):
    info = probe_video(tiny_video(tmp_path / "v.mp4", n_frames=20, fps=20.0))
    assert info["error"] == ""
    assert (info["width"], info["height"], info["n_frames"]) == (64, 48, 20)
    assert info["duration_s"] == pytest.approx(1.0)
    assert isinstance(pd.DataFrame([info]), pd.DataFrame)
