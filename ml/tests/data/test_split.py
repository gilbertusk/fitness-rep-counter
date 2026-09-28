import json

import numpy as np
import pandas as pd
import pytest

from repcount.data import split as split_mod
from repcount.data.extract import output_path
from repcount.data.split import (
    assign_class_groups,
    exceptions,
    load_videos,
    make_split,
    per_class_counts,
    render_report,
    split_document,
    split_targets,
)


def synthetic_videos(sizes: dict[str, int], dup_every: int = 4) -> pd.DataFrame:
    """Classes of the given size; every `dup_every`-th video duplicates the previous one (same group)."""
    rows, group = [], 0
    for label, n in sizes.items():
        for i in range(n):
            if i % dup_every != 1:
                group += 1
            rows.append({"video_id": f"{label}_{i}", "label": label, "group_id": f"g{group:04d}"})
    return pd.DataFrame(rows)


VIDEOS = synthetic_videos({"curl": 62, "press": 33, "squat": 20, "plank": 7, "twist": 13})


def test_no_group_is_in_two_splits():
    assignment = make_split(VIDEOS)
    per_group = VIDEOS.assign(split=VIDEOS["video_id"].map(assignment)).groupby("group_id")["split"].nunique()
    assert (per_group == 1).all()


def test_every_video_gets_a_split():
    assignment = make_split(VIDEOS)
    assert set(assignment) == set(VIDEOS["video_id"])
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_proportions_are_close_to_target():
    counts = per_class_counts(VIDEOS, make_split(VIDEOS)).sum()
    assert counts["train"] / counts["total"] == pytest.approx(0.70, abs=0.04)
    assert counts["val"] / counts["total"] == pytest.approx(0.15, abs=0.04)
    assert counts["test"] / counts["total"] == pytest.approx(0.15, abs=0.04)


def test_split_is_stratified_per_class():
    big = per_class_counts(VIDEOS, make_split(VIDEOS)).loc["curl"]
    assert abs(big["val"] - 9) <= 1 and abs(big["test"] - 9) <= 1


def test_small_classes_get_at_least_one_val_and_test_video():
    counts = per_class_counts(VIDEOS, make_split(VIDEOS))
    assert (counts[["val", "test"]] >= 1).all().all()


def test_split_is_deterministic_for_a_seed():
    assert make_split(VIDEOS, seed=42) == make_split(VIDEOS, seed=42)
    assert make_split(VIDEOS, seed=42) != make_split(VIDEOS, seed=7)


@pytest.mark.parametrize(("n", "expected"), [(62, (44, 9, 9)), (7, (5, 1, 1)), (3, (1, 1, 1)), (2, (2, 0, 0))])
def test_split_targets(n, expected):
    t = split_targets(n)
    assert (t["train"], t["val"], t["test"]) == expected


def test_assign_class_groups_keeps_big_groups_whole():
    groups = {"a": 5, "b": 1, "c": 1, "d": 1}
    assignment = assign_class_groups(groups, {"train": 5, "val": 2, "test": 1}, np.random.default_rng(0))
    assert assignment["a"] == "train"
    assert sorted(assignment[g] for g in "bcd") == ["test", "val", "val"]


def test_exceptions_flag_small_and_empty_classes():
    counts = pd.DataFrame({"train": [5, 2], "val": [1, 0], "test": [1, 0], "total": [7, 2]}, index=["plank", "x"])
    notes = exceptions(counts)
    assert any(n.startswith("plank: hanya 7 video") for n in notes)
    assert "x: val atau test kosong" in notes


def write_fixture(tmp_path, skip: str | None = None) -> None:
    VIDEOS.assign(path="p").to_csv(tmp_path / "manifest.csv", index=False)
    for row in VIDEOS.itertuples():
        if row.video_id != skip:
            out = output_path(tmp_path / "kp", row.label, row.video_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"")


def test_document_and_report_from_files(tmp_path):
    write_fixture(tmp_path, skip="plank_0")

    videos, excluded = load_videos(tmp_path / "manifest.csv", tmp_path / "kp")
    doc = split_document(videos, excluded, make_split(videos), 42)
    report = render_report(doc, videos, excluded)

    assert doc["excluded"] == ["plank_0"]
    assert sum(len(v) for v in doc["splits"].values()) == len(VIDEOS) - 1
    assert json.loads(json.dumps(doc)) == doc
    assert "| curl |" in report and "plank_0" in report


def test_main_writes_json_and_report(tmp_path, monkeypatch):
    write_fixture(tmp_path)
    monkeypatch.setattr(split_mod.config, "MANIFEST_PATH", tmp_path / "manifest.csv")
    monkeypatch.setattr(split_mod.config, "KEYPOINTS_DIR", tmp_path / "kp")
    monkeypatch.setattr("sys.argv", ["split", "--out", str(tmp_path / "s.json"), "--report", str(tmp_path / "s.md")])

    split_mod.main()

    doc = json.loads((tmp_path / "s.json").read_text())
    assert doc["excluded"] == [] and sum(len(v) for v in doc["splits"].values()) == len(VIDEOS)
    assert (tmp_path / "s.md").read_text(encoding="utf-8").startswith("# Split data v1")
