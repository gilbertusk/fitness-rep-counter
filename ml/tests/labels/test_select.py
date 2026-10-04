"""Unit tests for repcount.labels.select."""

import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

from repcount.labels import select


def manifest_rows(label: str, n: int, ext: str = "mp4", duration: float = 5.0, group_from: int = 0) -> list[dict]:
    return [{
        "video_id": f"{label}_{i}", "label": label, "path": f"{label}/{label}_{i}.{ext}", "ext": ext,
        "fps": 30.0, "duration_s": duration, "group_id": f"{label}_g{group_from + i}", "error": "",
    } for i in range(n)]


def split_doc(val: list[str], test: list[str], train: list[str] | None = None) -> dict:
    return {"splits": {"train": train or [], "val": val, "test": test}}


def pool_of(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.assign(split="val", detected_pct=math.nan)


# ---------------------------------------------------------------- candidates


def test_only_val_and_test_videos_are_candidates():
    manifest = pd.DataFrame(manifest_rows("squat", 6))
    pool = select.candidates(manifest, split_doc(val=["squat_0"], test=["squat_1"], train=["squat_2", "squat_3"]))
    assert sorted(pool["video_id"]) == ["squat_0", "squat_1"]
    assert dict(zip(pool["video_id"], pool["split"], strict=True)) == {"squat_0": "val", "squat_1": "test"}


def test_a_video_that_failed_to_open_is_never_a_candidate():
    rows = manifest_rows("squat", 2)
    rows[1]["error"] = "cannot decode first frame"
    pool = select.candidates(pd.DataFrame(rows), split_doc(val=["squat_0", "squat_1"], test=[]))
    assert pool["video_id"].tolist() == ["squat_0"]


def test_plank_is_a_hold_and_everything_else_is_reps():
    assert select.task_for("plank") == "hold"
    assert select.task_for("squat") == "reps"


# ---------------------------------------------------------------- variety


def test_at_most_k_videos_are_picked_per_class():
    assert len(select.pick_diverse(pool_of(manifest_rows("squat", 12)), k=5)) == 5


def test_a_small_class_gives_everything_it_has():
    assert sorted(select.pick_diverse(pool_of(manifest_rows("plank", 2)), k=5)) == ["plank_0", "plank_1"]


def test_both_file_extensions_are_represented_when_the_class_has_them():
    rows = manifest_rows("squat", 8, ext="mp4") + [
        {**r, "video_id": f"squat_mov_{i}", "ext": "mov", "group_id": f"mov_g{i}"}
        for i, r in enumerate(manifest_rows("squat", 2, ext="mov"))
    ]
    picked = select.pick_diverse(pool_of(rows), k=3)
    assert any(v.startswith("squat_mov") for v in picked)
    assert any(not v.startswith("squat_mov") for v in picked)


def test_near_duplicates_are_avoided_while_other_groups_remain():
    rows = manifest_rows("squat", 6)
    for row in rows[:5]:
        row["group_id"] = "same_source"  # five clips cut from one video
    picked = select.pick_diverse(pool_of(rows), k=2)
    groups = {r["video_id"]: r["group_id"] for r in rows}
    assert len({groups[v] for v in picked}) == 2


def test_durations_are_spread_rather_than_clustered():
    rows = manifest_rows("squat", 9)
    for row, duration in zip(rows, [2, 2.1, 2.2, 5, 5.1, 5.2, 20, 20.5, 21], strict=True):
        row["duration_s"] = duration
    picked = select.pick_diverse(pool_of(rows), k=3)
    durations = sorted(next(r["duration_s"] for r in rows if r["video_id"] == v) for v in picked)
    assert durations[0] < 3 and 4 < durations[1] < 6 and durations[2] > 19


def test_the_pick_is_reproducible_for_a_seed():
    pool = pool_of(manifest_rows("squat", 12))
    assert select.pick_diverse(pool, 5, seed=42) == select.pick_diverse(pool, 5, seed=42)


def test_one_class_does_not_change_when_another_is_added():
    """Each class has its own generator, so the list stays stable as the dataset grows."""
    squat = manifest_rows("squat", 12)
    alone = select.select_videos(pool_of(squat))
    together = select.select_videos(pool_of(squat + manifest_rows("lunge", 12)))
    assert alone["video_id"].tolist() == together[together["label"] == "squat"]["video_id"].tolist()


def test_unknown_durations_do_not_break_the_pick():
    rows = manifest_rows("squat", 6)
    rows[2]["duration_s"] = None
    assert len(select.pick_diverse(pool_of(rows), k=5)) == 5


def test_an_empty_class_picks_nothing():
    assert select.pick_diverse(pool_of(manifest_rows("squat", 3)).iloc[0:0], k=5) == []
    assert select.pick_diverse(pool_of(manifest_rows("squat", 3)), k=0) == []


# ---------------------------------------------------------------- output


def test_selection_has_the_to_label_layout_sorted_by_class():
    chosen = select.select_videos(pool_of(manifest_rows("squat", 7) + manifest_rows("plank", 2)))
    assert list(chosen.columns) == select.TO_LABEL_COLUMNS
    assert chosen["label"].tolist() == ["plank"] * 2 + ["squat"] * 5
    assert set(chosen.loc[chosen["label"] == "plank", "task"]) == {"hold"}


def test_videos_without_any_pose_are_dropped_but_unknown_ones_are_kept():
    pool = pool_of(manifest_rows("squat", 3)).assign(detected_pct=[0.0, 55.0, math.nan])
    kept, dropped = select.drop_poseless(pool)
    assert dropped["video_id"].tolist() == ["squat_0"]
    assert kept["video_id"].tolist() == ["squat_1", "squat_2"]


def test_shortfalls_list_the_classes_that_could_not_fill_their_quota():
    pool = pool_of(manifest_rows("squat", 9) + manifest_rows("plank", 2))
    chosen = select.select_videos(pool)
    assert select.shortfalls(pool, chosen) == {"plank": 2}


# ---------------------------------------------------------------- I/O


def write_npz(path, n_frames: int, n_detected: int) -> None:
    landmarks = np.full((n_frames, 33, 4), np.nan, np.float32)
    landmarks[:n_detected] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, landmarks=landmarks)


def test_detection_rate_is_the_share_of_frames_with_a_pose(tmp_path):
    write_npz(tmp_path / "a.npz", n_frames=10, n_detected=4)
    assert select.detection_rate(tmp_path / "a.npz") == pytest.approx(40.0)
    write_npz(tmp_path / "b.npz", n_frames=0, n_detected=0)
    assert select.detection_rate(tmp_path / "b.npz") == 0.0


def test_detection_rates_come_from_the_keypoints_or_stay_unknown(tmp_path):
    write_npz(tmp_path / "squat" / "squat_0.npz", n_frames=10, n_detected=10)
    pool = pool_of(manifest_rows("squat", 2))
    rates = select.attach_detection_rates(pool, tmp_path)["detected_pct"].tolist()
    assert rates[0] == 100.0 and math.isnan(rates[1])


def test_the_cli_writes_to_label_csv_and_reports_what_it_left_out(tmp_path, monkeypatch, capsys):
    rows = manifest_rows("squat", 8) + manifest_rows("plank", 2)
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    ids = [r["video_id"] for r in rows]
    (tmp_path / "split.json").write_text(json.dumps(split_doc(val=ids[:5] + ids[8:], test=ids[5:8])))
    write_npz(tmp_path / "kp" / "squat" / "squat_0.npz", n_frames=10, n_detected=0)

    out = tmp_path / "to_label.csv"
    monkeypatch.setattr(sys, "argv", ["prog", "--manifest", str(tmp_path / "manifest.csv"),
                                      "--split-file", str(tmp_path / "split.json"),
                                      "--keypoints-dir", str(tmp_path / "kp"), "--out", str(out)])
    select.main()

    written = pd.read_csv(out)
    assert list(written.columns) == select.TO_LABEL_COLUMNS
    assert "squat_0" not in written["video_id"].tolist()
    printed = capsys.readouterr().out
    assert "left out (no pose in any frame): squat_0" in printed
    assert "only 2 available for plank" in printed


def test_the_cli_explains_a_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--manifest", str(tmp_path / "nope.csv")])
    with pytest.raises(SystemExit, match="run stage 0 first"):
        select.main()
