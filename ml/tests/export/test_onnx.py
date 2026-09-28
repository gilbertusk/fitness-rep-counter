"""Unit tests for repcount.export.onnx.

These run on a randomly initialised model: they check the export path, not accuracy.
"""

import json
import shutil

import numpy as np
import pytest

from repcount.export import onnx as export
from repcount.features import features as feat
from repcount.models.temporal import build_model

N_CLASSES = 22


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export once; the ONNX conversion is the slow part of this module."""
    directory = tmp_path_factory.mktemp("export")
    model = build_model(N_CLASSES)
    path = export.export_model(model, directory / "exercise_classifier.onnx")
    return {"model": model, "path": path, "dir": directory}


def test_the_export_is_a_single_self_contained_file(exported):
    """The exporter defaults to spilling weights into a sibling .data file; app/models/ must not."""
    assert not exported["path"].with_suffix(".onnx.data").exists()
    assert list(exported["dir"].iterdir()) == [exported["path"]]


def test_the_weights_really_are_inside_the_file(exported):
    import onnx as onnx_module

    model = onnx_module.load(str(exported["path"]), load_external_data=False)
    external = [
        initializer.name for initializer in model.graph.initializer
        if initializer.HasField("data_location") and initializer.data_location == onnx_module.TensorProto.EXTERNAL
    ]
    assert external == []
    stored = sum(int(np.prod(i.dims)) for i in model.graph.initializer)
    assert stored * 4 < exported["path"].stat().st_size, "the file must be at least as big as its weights"


def test_the_exported_opset_is_the_one_recorded_in_the_report(exported):
    import onnx as onnx_module

    model = onnx_module.load(str(exported["path"]), load_external_data=False)
    versions = {opset.domain or "ai.onnx": opset.version for opset in model.opset_import}
    assert versions["ai.onnx"] == export.OPSET


def test_the_model_loads_on_its_own_away_from_the_export_directory(exported, tmp_path):
    import onnxruntime

    solo = tmp_path / "moved.onnx"
    shutil.copy(exported["path"], solo)
    session = onnxruntime.InferenceSession(str(solo), providers=["CPUExecutionProvider"])
    window = np.zeros((1, feat.WINDOW_FRAMES, feat.N_FEATURES), np.float32)
    assert session.run([export.OUTPUT_NAME], {export.INPUT_NAME: window})[0].shape == (1, N_CLASSES)


def test_the_batch_axis_stays_dynamic(exported):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(exported["path"]), providers=["CPUExecutionProvider"])
    for batch in (1, 4):
        window = np.zeros((batch, feat.WINDOW_FRAMES, feat.N_FEATURES), np.float32)
        assert session.run([export.OUTPUT_NAME], {export.INPUT_NAME: window})[0].shape == (batch, N_CLASSES)


def test_onnx_agrees_with_pytorch(exported):
    result = export.verify_export(exported["model"], exported["path"], n_windows=20)
    assert result["n_windows"] == 20
    assert result["max_abs_difference"] < export.TOLERANCE


def test_latency_is_measured_and_finite(exported):
    latency = export.measure_latency(exported["path"], n_runs=20)
    assert latency["n_runs"] == 20
    assert 0 < latency["median_ms"] < 1000
    assert latency["p95_ms"] >= latency["median_ms"]


# ---------------------------------------------------------------- metadata


def test_labels_document_tells_the_browser_what_it_needs():
    document = export.labels_document(
        ["squat", "push_up"], {"minConfidence": 0.6}, {"max_abs_difference": 1e-8}, {"median_ms": 0.1}, 1234,
    )
    assert document["labels"] == ["squat", "push_up"]
    assert document["thresholds"] == {"minConfidence": 0.6}
    assert document["window"] == {
        "frames": feat.WINDOW_FRAMES, "stride": feat.WINDOW_STRIDE,
        "features": feat.N_FEATURES, "targetFps": feat.TARGET_FPS, "spec": "docs/FEATURES.md",
    }
    assert document["export"]["modelBytes"] == 1234
    assert document["export"]["opset"] == export.OPSET


def test_labels_document_leaves_thresholds_empty_until_they_are_measured():
    """classifier.js keeps its own provisional defaults rather than inventing measured ones."""
    assert export.labels_document(["a"], None, {}, {}, 0)["thresholds"] == {}


def test_the_document_survives_a_json_round_trip():
    document = export.labels_document(["a", "b"], {"minMargin": 0.2}, {"n_windows": 100}, {"p95_ms": 0.3}, 99)
    assert json.loads(json.dumps(document)) == document


def test_reading_a_run_without_a_checkpoint_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="train a model first"):
        export.read_run_metadata(tmp_path)


def test_reading_a_run_returns_its_metadata(tmp_path):
    (tmp_path / "temporal.json").write_text(json.dumps({"labels": ["a", "b"], "val_macro_f1": 0.5}))
    assert export.read_run_metadata(tmp_path)["labels"] == ["a", "b"]


def test_a_checkpoint_round_trips_through_load_temporal(tmp_path):
    import torch

    labels = ["a", "b", "c"]
    original = build_model(len(labels))
    torch.save(original.state_dict(), tmp_path / "temporal.pt")

    loaded = export.load_temporal(tmp_path, labels)
    window = torch.zeros(1, feat.WINDOW_FRAMES, feat.N_FEATURES)
    original.eval()
    with torch.no_grad():
        assert torch.allclose(loaded(window), original(window))
