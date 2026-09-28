import numpy as np
import pandas as pd
import pytest

from repcount.evaluation.pose_quality import (
    aggregate_by_class,
    dominant_side,
    findings,
    load_video_stats,
    render_markdown,
    side_summary,
    video_quality,
)

LEFT = [11, 13, 15, 23, 25, 27]
RIGHT = [12, 14, 16, 24, 26, 28]


def landmarks(n_frames: int, n_missing: int, left_vis: float, right_vis: float) -> np.ndarray:
    arr = np.full((n_frames, 33, 4), 0.5, np.float32)
    arr[:, LEFT, 3] = left_vis
    arr[:, RIGHT, 3] = right_vis
    arr[:n_missing] = np.nan
    return arr


def test_video_quality_counts_detected_frames_and_ignores_nan():
    stats = video_quality(landmarks(10, 4, left_vis=0.9, right_vis=0.3))
    assert stats["n_frames"] == 10 and stats["n_detected"] == 6
    assert stats["vis_elbow"] == pytest.approx(0.6)
    assert stats["side"] == "left"


def test_video_without_any_pose_has_nan_visibility():
    stats = video_quality(landmarks(5, 5, 0.9, 0.9))
    assert stats["n_detected"] == 0 and np.isnan(stats["vis_knee"]) and stats["side"] == "unknown"


@pytest.mark.parametrize(("left", "right", "side"), [(0.9, 0.5, "left"), (0.5, 0.9, "right"), (0.8, 0.82, "balanced")])
def test_dominant_side(left, right, side):
    assert dominant_side(left, right) == side


def test_side_summary_reports_majority_share():
    assert side_summary(pd.Series(["left", "left", "right", "unknown"])) == "left (67%)"
    assert side_summary(pd.Series(["unknown"])) == "unknown"


def per_video_rows() -> pd.DataFrame:
    rows = []
    for label, vid, arr in [
        ("bench", "b1", landmarks(10, 8, 0.4, 0.2)),
        ("bench", "b2", landmarks(10, 10, 0.4, 0.2)),
        ("squat", "s1", landmarks(30, 0, 0.9, 0.9)),
        ("squat", "s2", landmarks(10, 0, 0.5, 0.5)),
    ]:
        rows.append({"video_id": vid, "label": label, "duration_s": len(arr) / 30, **video_quality(arr)})
    return pd.DataFrame(rows)


def test_aggregate_by_class_sorts_worst_first_and_weights_by_frames():
    table = aggregate_by_class(per_video_rows())
    assert table["label"].tolist() == ["bench", "squat"]
    bench, squat = table.iloc[0], table.iloc[1]
    assert bench["detected_pct"] == pytest.approx(10.0)
    assert squat["detected_pct"] == pytest.approx(100.0)
    assert squat["vis_knee"] == pytest.approx((30 * 0.9 + 10 * 0.5) / 40)
    assert bench["n_videos"] == 2 and bench["side"] == "left (100%)"


def test_findings_and_markdown_mention_worst_class():
    videos = per_video_rows()
    table = aggregate_by_class(videos)
    notes = findings(table, videos)
    md = render_markdown(table, videos, n_missing=1)
    assert 3 <= len(notes) <= 5
    assert "**bench** (10.0%)" in notes[0] and "b2" in notes[-1]
    assert md.index("| bench |") < md.index("| squat |")


def test_load_video_stats_reads_npz_files(tmp_path):
    out = tmp_path / "squat" / "s1.npz"
    out.parent.mkdir()
    np.savez_compressed(out, landmarks=landmarks(30, 3, 0.9, 0.9), fps_effective=np.float32(30))
    manifest = pd.DataFrame({"video_id": ["s1", "s2"], "label": ["squat", "squat"]})

    stats = load_video_stats(manifest, tmp_path)

    assert stats["video_id"].tolist() == ["s1"]
    assert stats.loc[0, "duration_s"] == pytest.approx(1.0) and stats.loc[0, "n_detected"] == 27
