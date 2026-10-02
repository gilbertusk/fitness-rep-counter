"""Unit tests for repcount.evaluation.classifier_report."""

import numpy as np
import pytest

from repcount.evaluation import classifier_report as report

LABELS = ["squat", "push_up", "plank"]


# ---------------------------------------------------------------- aggregation


def test_video_level_prediction_averages_the_windows_of_that_video():
    probabilities = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],   # squat_1: mean is [0.5, 0.5, 0.0]
        [0.2, 0.3, 0.5],
    ])
    video_ids = np.array(["squat_1", "squat_1", "plank_2"], dtype=object)
    averaged, order = report.aggregate_by_video(probabilities, video_ids)

    assert order.tolist() == ["plank_2", "squat_1"]
    assert averaged[order.tolist().index("squat_1")] == pytest.approx([0.5, 0.5, 0.0])
    assert averaged[order.tolist().index("plank_2")] == pytest.approx([0.2, 0.3, 0.5])


def test_a_single_odd_window_does_not_decide_the_video():
    """Nine confident windows outvote one wrong one once they are averaged."""
    probabilities = np.array([[0.8, 0.2, 0.0]] * 9 + [[0.0, 1.0, 0.0]])
    video_ids = np.array(["v_1"] * 10, dtype=object)
    averaged, _ = report.aggregate_by_video(probabilities, video_ids)
    assert averaged.argmax(axis=1).tolist() == [0]


def test_video_truth_takes_the_label_of_the_videos_windows():
    y = np.array([0, 0, 2])
    video_ids = np.array(["a_1", "a_1", "c_2"], dtype=object)
    assert report.video_truth(y, video_ids).tolist() == [0, 2]  # sorted by video id: a_1, c_2


# ---------------------------------------------------------------- metrics


def test_metrics_on_a_perfect_prediction():
    y = np.array([0, 1, 2, 0])
    result = report.metrics(y, y, LABELS)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["n"] == 4
    assert set(result["per_class_f1"]) == set(LABELS)


def test_macro_f1_punishes_ignoring_a_rare_class():
    """Nine of ten right, but the rare class is never predicted: accuracy stays high, macro-F1 drops."""
    y_true = np.array([0] * 9 + [2])
    y_pred = np.array([0] * 10)
    result = report.metrics(y_true, y_pred, LABELS)
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["macro_f1"] < 0.4
    assert result["per_class_f1"]["plank"] == 0.0


def test_per_class_f1_covers_every_label_even_when_absent_from_the_split():
    result = report.metrics(np.array([0, 0]), np.array([0, 0]), LABELS)
    assert result["per_class_f1"]["plank"] == 0.0
    assert len(result["per_class_f1"]) == len(LABELS)


# ---------------------------------------------------------------- confusions


def test_confused_pairs_ranks_the_most_frequent_mistake_first():
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([1, 1, 0, 1, 0])
    pairs = report.confused_pairs(y_true, y_pred, LABELS)

    assert pairs[0]["true"] == "squat" and pairs[0]["predicted"] == "push_up"
    assert pairs[0]["count"] == 2
    assert pairs[0]["share_of_true"] == pytest.approx(2 / 3)


def test_confused_pairs_ignores_correct_predictions_and_respects_top_n():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    assert report.confused_pairs(y_true, y_pred, LABELS) == []
    y_pred = np.array([1, 2, 0])
    assert len(report.confused_pairs(y_true, y_pred, LABELS, top_n=2)) == 2


# ---------------------------------------------------------------- thresholds


def test_the_sweep_picks_a_cut_off_that_reaches_the_accuracy_target():
    # Confident rows are right, unconfident rows are wrong: a cut-off exists that separates them.
    confident = np.tile([0.95, 0.03, 0.02], (20, 1))
    unsure = np.tile([0.4, 0.35, 0.25], (20, 1))
    probabilities = np.vstack([confident, unsure])
    y = np.array([0] * 20 + [1] * 20)

    chosen = report.sweep_threshold(probabilities, y, "confidence", target=0.9)
    assert 0.4 < chosen["threshold"] <= 0.95
    assert chosen["accuracy_when_kept"] >= 0.9
    assert 0 < chosen["coverage"] <= 1


def test_the_sweep_prefers_the_lowest_cut_off_that_works_so_coverage_stays_high():
    probabilities = np.tile([0.55, 0.25, 0.20], (10, 1))
    y = np.zeros(10, dtype=int)
    chosen = report.sweep_threshold(probabilities, y, "confidence", target=0.9)
    assert chosen["threshold"] == 0.0
    assert chosen["coverage"] == 1.0


def test_the_sweep_says_so_when_the_target_is_unreachable():
    probabilities = np.tile([0.9, 0.05, 0.05], (10, 1))
    y = np.ones(10, dtype=int)  # always wrong, however confident
    chosen = report.sweep_threshold(probabilities, y, "confidence", target=0.9)
    assert "note" in chosen
    assert chosen["accuracy_when_kept"] == 0.0


def test_the_margin_sweep_uses_the_gap_between_the_top_two():
    close = np.tile([0.50, 0.48, 0.02], (10, 1))   # tiny margin, wrong
    clear = np.tile([0.90, 0.05, 0.05], (10, 1))   # big margin, right
    probabilities = np.vstack([close, clear])
    y = np.array([1] * 10 + [0] * 10)
    chosen = report.sweep_threshold(probabilities, y, "margin", target=0.9)
    assert chosen["kind"] == "margin"
    assert chosen["threshold"] > 0.02
    assert chosen["accuracy_when_kept"] >= 0.9


# ---------------------------------------------------------------- report output


def _results(window_f1: float, video_f1: float) -> dict:
    metrics = {"accuracy": video_f1, "macro_f1": video_f1, "n": 10,
               "per_class_f1": dict.fromkeys(LABELS, video_f1)}
    return {
        "baseline (gradient boosting)": {
            "window": {**metrics, "macro_f1": window_f1 - 0.1},
            "video": {**metrics, "macro_f1": video_f1 - 0.1},
        },
        "temporal (1D-CNN)": {"window": {**metrics, "macro_f1": window_f1}, "video": {**metrics}},
    }


def test_the_report_states_the_test_set_is_used_once_and_names_the_best_model():
    thresholds = {
        "confidence": {"kind": "confidence", "threshold": 0.6, "coverage": 0.8, "accuracy_when_kept": 0.92},
        "margin": {"kind": "margin", "threshold": 0.15, "coverage": 0.7, "accuracy_when_kept": 0.91},
    }
    pairs = [{"true": "squat", "predicted": "push_up", "count": 3, "share_of_true": 0.3}]
    counts = {"train": 100, "val": 20, "test": 20}

    text = report.render_report(_results(0.7, 0.8), LABELS, thresholds, pairs, counts, None)

    assert "Test set dipakai sekali saja" in text
    assert "temporal (1D-CNN)" in text
    assert "Level window" in text and "Level video" in text
    assert "| squat | push_up | 3 | 30% |" in text
    assert all(label in text for label in LABELS)
    assert "figures/confusion_matrix.png" in text


def test_the_report_includes_the_browser_numbers_once_the_model_is_exported():
    export = {
        "modelBytes": 276701, "opset": 18,
        "verification": {"n_windows": 100, "max_abs_difference": 4.47e-08, "tolerance": 1e-4},
        "latencyCpu": {"median_ms": 0.08, "p95_ms": 0.11, "n_runs": 200},
    }
    thresholds = {"confidence": {"kind": "confidence", "threshold": 0.6, "coverage": 0.8,
                                 "accuracy_when_kept": 0.92}}
    text = report.render_report(_results(0.7, 0.8), LABELS, thresholds, [], {"train": 1, "val": 1, "test": 1}, export)

    assert "270 KB" in text
    assert "opset 18" in text
    assert "0.08 ms" in text


def test_the_confusion_matrix_figure_is_written(tmp_path):
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    path = report.plot_confusion_matrix(y_true, y_pred, LABELS, tmp_path / "figures" / "confusion_matrix.png")
    assert path.exists()
    assert path.stat().st_size > 1000


# ---------------------------------------------------------------- end to end


def test_evaluate_all_reports_both_levels_for_every_model():
    rng = np.random.default_rng(0)
    n = 12
    data = {
        "labels": LABELS,
        "val": {"x": rng.normal(0, 1, (n, 30, 47)), "y": rng.integers(0, 3, n)},
        "test": {
            "x": rng.normal(0, 1, (n, 30, 47)),
            "y": np.repeat([0, 1, 2], 4),
            "video_ids": np.array(np.repeat(["a_1", "b_2", "c_3"], 4), dtype=object),
        },
    }

    def fake_predict(_model, x):
        probabilities = rng.random((len(x), len(LABELS)))
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    results, validation = report.evaluate_all(data, {"fake": (fake_predict, None)})
    assert set(results["fake"]) == {"window", "video", "video_predictions", "video_ids"}
    assert results["fake"]["video"]["n"] == 3, "three videos, whatever the window count"
    assert results["fake"]["window"]["n"] == n
    assert validation["fake"].shape == (n, len(LABELS))


def test_metrics_count_the_test_videos_of_every_class():
    result = report.metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), ["a", "b", "c"])
    assert result["per_class_support"] == {"a": 2, "b": 1, "c": 0}


def test_pose_quality_is_read_per_class_with_the_dataset_wide_share(tmp_path):
    path = tmp_path / "pose_quality.csv"
    path.write_text("label,n_frames,n_detected,detected_pct\na,100,90,90.0\nb,300,300,100.0\n", encoding="utf-8")
    detection = report.read_pose_detection(path)
    assert detection["a"] == 90.0 and detection["b"] == 100.0
    assert detection["__all__"] == 97.5
    assert report.read_pose_detection(tmp_path / "missing.csv") is None


def test_pose_context_only_blames_pose_quality_where_the_numbers_support_it():
    detection = {"curl": 99.0, "decline": 87.6, "__all__": 97.7}
    text = report.pose_context([("curl", 0.5), ("decline", 0.57)], detection)[0]
    assert "curl 99.0%, decline 87.6%" in text
    assert "bisa ikut menjelaskan: decline." in text
    assert "**tidak** menjelaskan: curl" in text
    assert "tidak tersedia" in report.pose_context([("curl", 0.5)], None)[0]


def test_the_report_shows_test_support_per_class_and_warns_about_tiny_classes():
    results = _results(0.7, 0.8)
    results["temporal (1D-CNN)"]["video"]["per_class_support"] = {label: 1 for label in LABELS}
    text = report.render_report(results, LABELS, {}, [], {"train": 1, "val": 1, "test": 1}, None,
                                {label: 99.0 for label in LABELS} | {"__all__": 97.7})
    assert "| n video test |" in text
    assert f"| {LABELS[0]} |" in text and "| 1 |" in text
    assert "jangan dibaca sebagai angka yang presisi" in text
    assert "Dugaan penyebab" not in text
