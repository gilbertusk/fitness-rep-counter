"""Unit tests for repcount.labels.validate."""

import sys

import pandas as pd
import pytest

from repcount.labels import validate

MANIFEST = pd.DataFrame([
    {"video_id": "squat_1", "label": "squat", "duration_s": 10.0},
    {"video_id": "squat_2", "label": "squat", "duration_s": 6.0},
    {"video_id": "plank_1", "label": "plank", "duration_s": 40.0},
])
MANIFEST_BY_ID = {r["video_id"]: r for r in MANIFEST.to_dict("records")}


def rep_row(**overrides) -> dict:
    return {
        "video_id": "squat_1", "label": "squat", "rep_count": "3", "rep_timestamps_ms": "2000;4500;7000",
        "hold_start_ms": "", "hold_end_ms": "", "is_ambiguous": "false", "notes": "",
        "labeler": "gk", "labeled_at": "2026-10-02T09:15:00.000Z", **overrides,
    }


def hold_row(**overrides) -> dict:
    return rep_row(**{"video_id": "plank_1", "label": "plank", "rep_count": "", "rep_timestamps_ms": "",
                      "hold_start_ms": "1500", "hold_end_ms": "38000", **overrides})


def problems_of(row: dict) -> list[str]:
    return validate.row_problems(row, MANIFEST_BY_ID.get(row["video_id"]))


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=validate.REP_LABEL_COLUMNS)


# ---------------------------------------------------------------- parsing


def test_timestamps_parse_from_the_semicolon_list():
    assert validate.parse_timestamps("1200;2400") == [1200, 2400]
    assert validate.parse_timestamps("") == []
    with pytest.raises(ValueError):
        validate.parse_timestamps("12.5;abc")


# ---------------------------------------------------------------- valid rows


def test_a_well_formed_rep_row_has_no_problems():
    assert problems_of(rep_row()) == []


def test_a_well_formed_hold_row_has_no_problems():
    assert problems_of(hold_row()) == []


def test_an_ambiguous_row_with_a_note_is_valid():
    assert problems_of(rep_row(is_ambiguous="true", notes="dua orang di frame")) == []


def test_a_mark_on_the_last_frame_survives_a_slightly_shorter_manifest_duration():
    """The browser and OpenCV can disagree on duration by a frame; the last mark must not fail for that."""
    assert problems_of(rep_row(rep_count="1", rep_timestamps_ms="10100")) == []


# ---------------------------------------------------------------- broken rules


@pytest.mark.parametrize(("overrides", "expected"), [
    ({"video_id": "ghost_9"}, "tidak ada di manifest"),
    ({"label": "push_up"}, "≠ manifest"),
    ({"rep_count": "2"}, "≠ jumlah timestamp"),
    ({"rep_timestamps_ms": "2000;2000;7000"}, "tidak naik atau ada duplikat"),
    ({"rep_timestamps_ms": "4500;2000;7000"}, "tidak naik atau ada duplikat"),
    ({"rep_timestamps_ms": "2000;4500;99000"}, "di luar durasi"),
    ({"rep_timestamps_ms": "-5;4500;7000"}, "di luar durasi"),
    ({"rep_timestamps_ms": "2000;x;7000"}, "bukan bilangan bulat"),
    ({"hold_start_ms": "100"}, "harus kosong untuk latihan repetisi"),
    ({"is_ambiguous": "yes"}, "harus 'true' atau 'false'"),
    ({"is_ambiguous": "true"}, "ambigu tanpa catatan"),
    ({"labeler": " "}, "labeler kosong"),
    ({"labeled_at": "kemarin"}, "bukan waktu ISO 8601"),
    ({"rep_count": "0", "rep_timestamps_ms": ""}, "0 rep tanpa ditandai ambigu"),
])
def test_each_rule_in_the_readme_is_enforced_for_reps(overrides, expected):
    assert any(expected in p for p in problems_of(rep_row(**overrides))), problems_of(rep_row(**overrides))


@pytest.mark.parametrize(("overrides", "expected"), [
    ({"rep_count": "1", "rep_timestamps_ms": "5000"}, "harus kosong"),
    ({"hold_start_ms": "9000", "hold_end_ms": "3000"}, "mulai < selesai"),
    ({"hold_end_ms": "90000"}, "mulai < selesai"),
    ({"hold_end_ms": ""}, "wajib diisi"),
    ({"hold_start_ms": "abc"}, "bukan bilangan bulat"),
])
def test_each_rule_in_the_readme_is_enforced_for_holds(overrides, expected):
    assert any(expected in p for p in problems_of(hold_row(**overrides))), problems_of(hold_row(**overrides))


def test_an_ambiguous_plank_may_leave_the_hold_empty():
    assert problems_of(hold_row(hold_start_ms="", hold_end_ms="", is_ambiguous="true",
                                notes="tidak pernah mencapai posisi plank")) == []


def test_an_ambiguous_video_may_have_zero_reps():
    assert problems_of(rep_row(rep_count="0", rep_timestamps_ms="", is_ambiguous="true",
                               notes="bukan squat")) == []


# ---------------------------------------------------------------- whole file


def test_duplicate_rows_and_missing_columns_are_reported():
    assert ("squat_1", "video_id muncul lebih dari sekali") in validate.validate_labels(
        frame(rep_row(), rep_row()), MANIFEST)
    problems = validate.validate_labels(pd.DataFrame([{"video_id": "squat_1"}]), MANIFEST)
    assert problems[0][1].startswith("kolom hilang")


def test_a_clean_file_validates_to_nothing():
    assert validate.validate_labels(frame(rep_row(), hold_row()), MANIFEST) == []


def test_the_class_summary_counts_reps_and_hold_seconds():
    summary = validate.class_summary(frame(
        rep_row(), rep_row(video_id="squat_2", rep_count="1", rep_timestamps_ms="3000",
                           is_ambiguous="true", notes="x"),
        hold_row()))
    squat = summary[summary["label"] == "squat"].iloc[0]
    plank = summary[summary["label"] == "plank"].iloc[0]
    assert (squat["n_videos"], squat["n_ambiguous"], squat["mean"], squat["min"], squat["max"]) == (2, 1, 2.0, 1, 3)
    assert plank["unit"] == "detik tahan"
    assert plank["mean"] == pytest.approx(36.5)


def test_unlabelled_lists_what_is_left_in_to_label():
    to_label = pd.DataFrame({"video_id": ["squat_1", "squat_2", "plank_1"]})
    assert validate.unlabelled(frame(rep_row()), to_label) == ["plank_1", "squat_2"]


# ---------------------------------------------------------------- agreement


def test_marks_are_matched_one_to_one_within_the_tolerance():
    assert validate.matched_marks([1000, 2000, 3000], [1100, 2050, 3400]) == 2
    assert validate.matched_marks([1000, 1100], [1050]) == 1, "one mark cannot pair with two"
    assert validate.matched_marks([], [1000]) == 0


def test_agreement_reports_exact_off_by_one_mae_and_timing():
    first = frame(rep_row(), rep_row(video_id="squat_2", rep_count="2", rep_timestamps_ms="1000;3000"))
    second = frame(
        rep_row(rep_timestamps_ms="2100;4400;7050"),                                       # same count
        rep_row(video_id="squat_2", rep_count="3", rep_timestamps_ms="1000;3000;5000"),   # one more
    )
    result = validate.agreement(first, second)
    assert result["n_videos"] == 2
    assert result["exact"] == 0.5
    assert result["off_by_one"] == 1.0
    assert result["mae"] == 0.5
    assert result["timing_match"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_agreement_ignores_holds_and_videos_labelled_only_once():
    result = validate.agreement(frame(rep_row(), hold_row()), frame(hold_row()))
    assert result["n_videos"] == 0


# ---------------------------------------------------------------- CLI


@pytest.fixture
def files(tmp_path):
    MANIFEST.to_csv(tmp_path / "manifest.csv", index=False)
    frame(rep_row(), hold_row()).to_csv(tmp_path / "rep_labels.csv", index=False)
    pd.DataFrame({"video_id": ["squat_1", "squat_2", "plank_1"]}).to_csv(tmp_path / "to_label.csv", index=False)
    return tmp_path


def run(monkeypatch, files, *extra) -> int:
    monkeypatch.setattr(sys, "argv", ["prog", "--labels", str(files / "rep_labels.csv"),
                                      "--manifest", str(files / "manifest.csv"),
                                      "--to-label", str(files / "to_label.csv"), *extra])
    return validate.main()


def test_the_cli_passes_a_clean_file_and_prints_the_summary(files, monkeypatch, capsys):
    assert run(monkeypatch, files) == 0
    printed = capsys.readouterr().out
    assert "2 video berlabel" in printed
    assert "belum dilabel: 1 dari 3" in printed and "squat_2" in printed


def test_the_cli_fails_with_status_1_and_names_each_problem(files, monkeypatch, capsys):
    frame(rep_row(rep_count="5")).to_csv(files / "rep_labels.csv", index=False)
    assert run(monkeypatch, files) == 1
    assert "[rep_labels] squat_1: rep_count '5' ≠ jumlah timestamp (3)" in capsys.readouterr().out


def test_the_cli_compares_sessions_with_agreement(files, monkeypatch, capsys):
    frame(rep_row(rep_count="4", rep_timestamps_ms="2000;4500;7000;9000"), hold_row()).to_csv(
        files / "recheck.csv", index=False)
    assert run(monkeypatch, files, "--agreement", "--recheck", str(files / "recheck.csv")) == 0
    printed = capsys.readouterr().out
    assert "konsistensi antar sesi (1 video)" in printed
    assert "squat_1: 3 → 4 rep" in printed


def test_the_cli_checks_the_recheck_file_too(files, monkeypatch, capsys):
    frame(rep_row(labeler="")).to_csv(files / "recheck.csv", index=False)
    assert run(monkeypatch, files, "--agreement", "--recheck", str(files / "recheck.csv")) == 1
    assert "[recheck] squat_1: labeler kosong" in capsys.readouterr().out


def test_the_cli_explains_missing_files(files, monkeypatch):
    with pytest.raises(SystemExit, match="not found"):
        run(monkeypatch, files, "--agreement", "--recheck", str(files / "nope.csv"))
    (files / "rep_labels.csv").unlink()
    with pytest.raises(SystemExit, match="not found"):
        run(monkeypatch, files)
