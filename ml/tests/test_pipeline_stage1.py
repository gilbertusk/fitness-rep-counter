"""End-to-end test of the stage 1 command chain on synthetic keypoints.

Runs the four CLIs a contributor runs — windows → baseline → temporal → report → export — so the
wiring between them (run directories, label order, thresholds, the exported model) is covered even
without the 4.6 GB dataset. Accuracy is not asserted: three synthetic classes say nothing about 22
real ones. What is asserted is that each step produces the artefact the next one reads.
"""

import json
import sys

import numpy as np
import pytest

from repcount.features.golden import SYNTHETIC, synthetic_sequence

CLASSES = ("squat", "push_up", "plank")
VIDEOS_PER_CLASS = 4
N_FRAMES = 200


def clean_sequence(reps_per_second: float, seed: int) -> dict:
    """A gap-free, full-length synthetic clip whose speed identifies its class."""
    spec = {
        **SYNTHETIC,
        "n_frames": N_FRAMES,
        "reps_per_second": reps_per_second,
        "short_gap": (0, 0),
        "long_gap": (0, 0),
        "collapsed_scale": -1,
    }
    return synthetic_sequence(spec, seed=seed)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A keypoints directory and split file shaped exactly like the real pipeline's output."""
    root = tmp_path_factory.mktemp("stage1")
    keypoints = root / "keypoints"
    assignment = {"train": [], "val": [], "test": []}

    for label_index, label in enumerate(CLASSES):
        for n in range(VIDEOS_PER_CLASS):
            video_id = f"{label}_{n}"
            sequence = clean_sequence(0.5 * (label_index + 1), seed=100 * label_index + n)
            path = keypoints / label / f"{video_id}.npz"
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
            split = "train" if n < VIDEOS_PER_CLASS - 2 else ("val" if n == VIDEOS_PER_CLASS - 2 else "test")
            assignment[split].append(video_id)

    split_file = root / "split_test.json"
    split_file.write_text(json.dumps({"version": "test", "splits": assignment}), encoding="utf-8")
    return {"root": root, "keypoints": keypoints, "split_file": split_file, "run": root / "run"}


# The tests below run in definition order and build on each other: the run directory the
# baseline creates is the one the temporal model, the report and the export all read.
def run_cli(monkeypatch, main, *args) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", *args])
    main()


def test_windows_cli_reports_every_split(dataset, monkeypatch, capsys):
    from repcount.features.windows import main

    run_cli(monkeypatch, main, "--split-file", str(dataset["split_file"]),
            "--keypoints-dir", str(dataset["keypoints"]))
    output = capsys.readouterr().out
    assert f"{len(CLASSES)} classes" in output
    for split in ("train", "val", "test"):
        assert split in output


def test_the_split_is_read_per_video_with_no_leakage(dataset):
    from repcount.features.windows import load_all

    data = load_all(dataset["split_file"], dataset["keypoints"])
    assert data["labels"] == sorted(CLASSES)
    train_videos = set(data["train"]["video_ids"].tolist())
    for other in ("val", "test"):
        assert train_videos.isdisjoint(set(data[other]["video_ids"].tolist()))
    assert len(data["train"]["x"]) > 0 and len(data["test"]["x"]) > 0


def test_baseline_cli_writes_a_run(dataset, monkeypatch, capsys):
    from repcount.models.baseline import main

    run_cli(monkeypatch, main, "--split-file", str(dataset["split_file"]),
            "--keypoints-dir", str(dataset["keypoints"]), "--out", str(dataset["run"]))
    assert (dataset["run"] / "baseline.joblib").exists()
    metadata = json.loads((dataset["run"] / "baseline.json").read_text())
    assert metadata["labels"] == sorted(CLASSES)
    assert "val macro-F1" in capsys.readouterr().out


def test_temporal_cli_writes_a_checkpoint_the_exporter_can_read(dataset, monkeypatch, capsys):
    from repcount.models.temporal import main

    run_cli(monkeypatch, main, "--split-file", str(dataset["split_file"]),
            "--keypoints-dir", str(dataset["keypoints"]), "--out", str(dataset["run"]), "--epochs", "3")
    assert (dataset["run"] / "temporal.pt").exists()
    metadata = json.loads((dataset["run"] / "temporal.json").read_text())
    assert metadata["labels"] == sorted(CLASSES)
    assert metadata["n_parameters"] > 0
    assert "parameters" in capsys.readouterr().out


def test_report_cli_writes_the_markdown_figure_and_thresholds(dataset, monkeypatch, capsys):
    from repcount.evaluation.classifier_report import main

    out = dataset["root"] / "reports" / "classifier.md"
    run_cli(monkeypatch, main, "--run", str(dataset["run"]), "--split-file", str(dataset["split_file"]),
            "--keypoints-dir", str(dataset["keypoints"]), "--out", str(out))

    assert (out.parent / "figures" / "confusion_matrix.png").exists()
    text = out.read_text(encoding="utf-8")
    assert "Test set dipakai sekali saja" in text
    assert "Level window" in text and "Level video" in text
    assert "Ambang" in text
    for label in sorted(CLASSES):
        assert label in text

    thresholds = json.loads((dataset["run"] / "thresholds.json").read_text())
    assert 0.0 <= thresholds["minConfidence"] <= 1.0
    assert 0.0 <= thresholds["minMargin"] <= 1.0
    assert thresholds["chosenFrom"] == "validation split"
    assert "test macro-F1" in capsys.readouterr().out


def test_export_cli_ships_a_single_file_model_and_its_labels(dataset, monkeypatch, capsys):
    from repcount.export.onnx import main

    model_out = dataset["root"] / "models" / "exercise_classifier.onnx"
    labels_out = dataset["root"] / "models" / "labels.json"
    run_cli(monkeypatch, main, "--run", str(dataset["run"]),
            "--model-out", str(model_out), "--labels-out", str(labels_out))

    assert model_out.exists()
    assert not model_out.with_suffix(".onnx.data").exists()

    document = json.loads(labels_out.read_text())
    assert document["labels"] == sorted(CLASSES)
    # The report's thresholds must reach the browser, not classifier.js's placeholders.
    assert document["thresholds"]["minConfidence"] == \
        json.loads((dataset["run"] / "thresholds.json").read_text())["minConfidence"]
    assert document["export"]["modelBytes"] == model_out.stat().st_size
    assert document["export"]["verification"]["max_abs_difference"] < document["export"]["verification"]["tolerance"]
    assert document["window"]["spec"] == "docs/FEATURES.md"
    assert "ms/window" in capsys.readouterr().out


def test_the_exported_model_scores_a_window_produced_by_the_feature_pipeline(dataset):
    """The last link: features straight out of the spec go into the shipped model and come out scored."""
    import onnxruntime

    from repcount.export.onnx import INPUT_NAME, OUTPUT_NAME
    from repcount.features import features as feat

    sequence = clean_sequence(1.0, seed=999)
    features = feat.sequence_features(
        sequence["landmarks"], sequence["timestamps_ms"], sequence["width"], sequence["height"]
    )
    cut = feat.make_windows(features)
    window = cut["windows"][0].astype(np.float32)[None, :, :]

    session = onnxruntime.InferenceSession(str(dataset["root"] / "models" / "exercise_classifier.onnx"),
                                           providers=["CPUExecutionProvider"])
    scores = session.run([OUTPUT_NAME], {INPUT_NAME: window})[0]
    assert scores.shape == (1, len(CLASSES))
    assert np.isfinite(scores).all()
