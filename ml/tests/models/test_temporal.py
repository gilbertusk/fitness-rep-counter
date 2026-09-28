"""Unit tests for repcount.models.temporal."""

import numpy as np
import pytest

from repcount.features import features as feat
from repcount.models import temporal


def separable_dataset(n_per_class: int, n_classes: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Classes differ only in how fast their joint angles oscillate."""
    rng = np.random.default_rng(seed)
    t = np.arange(feat.WINDOW_FRAMES) / feat.TARGET_FPS
    windows, labels = [], []
    for label in range(n_classes):
        for _ in range(n_per_class):
            window = rng.normal(0, 0.05, (feat.WINDOW_FRAMES, feat.N_FEATURES))
            window[:, feat.N_POINTS * 3:] += np.sin(2 * np.pi * 0.5 * (label + 1) * t)[:, None]
            windows.append(window)
            labels.append(label)
    return np.array(windows, np.float32), np.array(labels, np.int64)


def ramp_window() -> np.ndarray:
    return np.tile(np.arange(feat.WINDOW_FRAMES, dtype=float)[:, None], (1, feat.N_FEATURES))


# ---------------------------------------------------------------- time scaling


def test_time_scale_of_one_is_the_identity():
    window = ramp_window()
    assert temporal.time_scale_window(window, 1.0) == pytest.approx(window)


def test_speeding_up_reaches_the_end_of_the_movement_sooner():
    scaled = temporal.time_scale_window(ramp_window(), 2.0)
    assert scaled[5, 0] == pytest.approx(10.0)
    assert scaled[-1, 0] == pytest.approx(feat.WINDOW_FRAMES - 1), "clipped at the last frame, never past it"


def test_slowing_down_stretches_the_movement():
    scaled = temporal.time_scale_window(ramp_window(), 0.5)
    assert scaled[10, 0] == pytest.approx(5.0)


def test_time_scaling_keeps_the_window_shape():
    assert temporal.time_scale_window(ramp_window(), 1.3).shape == (feat.WINDOW_FRAMES, feat.N_FEATURES)


# ---------------------------------------------------------------- augmentation


def test_augmentation_keeps_the_batch_shape():
    batch, _ = separable_dataset(4, n_classes=1)
    out = temporal.augment_batch(batch, np.random.default_rng(0))
    assert out.shape == batch.shape


def test_jitter_leaves_visibility_and_angles_alone():
    """Only x and y are spatial offsets; jittering a visibility or an angle would be meaningless."""
    batch = np.zeros((8, feat.WINDOW_FRAMES, feat.N_FEATURES))
    options = {**temporal.DEFAULTS, "flip_probability": 0.0, "time_scale": (1.0, 1.0)}
    out = temporal.augment_batch(batch, np.random.default_rng(0), options)

    visibility_columns = [i * 3 + 2 for i in range(feat.N_POINTS)]
    assert out[:, :, visibility_columns] == pytest.approx(0.0)
    assert out[:, :, feat.N_POINTS * 3:] == pytest.approx(0.0)
    position_columns = [i * 3 for i in range(feat.N_POINTS)]
    assert np.abs(out[:, :, position_columns]).max() > 0, "positions should be jittered"


def test_flipping_every_window_mirrors_the_batch():
    batch, _ = separable_dataset(4, n_classes=1, seed=5)
    options = {**temporal.DEFAULTS, "flip_probability": 1.0, "time_scale": (1.0, 1.0), "jitter_std": 0.0}
    out = temporal.augment_batch(batch, np.random.default_rng(0), options)
    assert out == pytest.approx(feat.flip_features(batch), abs=1e-9)


def test_augmentation_does_not_mutate_the_batch():
    batch, _ = separable_dataset(4, n_classes=1, seed=6)
    snapshot = batch.copy()
    temporal.augment_batch(batch, np.random.default_rng(0))
    assert batch == pytest.approx(snapshot)


# ---------------------------------------------------------------- class weights


def test_class_weights_are_inverse_frequency():
    weights = temporal.class_weights(np.array([0, 0, 0, 1]), n_classes=2)
    assert weights[1] / weights[0] == pytest.approx(3.0)


def test_a_class_absent_from_the_split_gets_weight_zero_not_infinity():
    weights = temporal.class_weights(np.array([0, 0, 1]), n_classes=4)
    assert weights[2] == 0.0 and weights[3] == 0.0
    assert np.isfinite(weights).all()


# ---------------------------------------------------------------- model


def test_the_model_fits_the_size_budget():
    model = temporal.build_model(22)
    parameters = temporal.count_parameters(model)
    assert parameters * 4 < 1_000_000, f"{parameters} parameters exceed the 1 MB float32 budget"


def test_the_model_maps_a_window_batch_to_one_score_per_class():
    import torch

    model = temporal.build_model(22)
    model.eval()
    with torch.no_grad():
        scores = model(torch.zeros(3, feat.WINDOW_FRAMES, feat.N_FEATURES))
    assert tuple(scores.shape) == (3, 22)


# ---------------------------------------------------------------- training


def test_training_learns_a_separable_dataset_and_reports_its_best_epoch():
    x_train, y_train = separable_dataset(20, seed=1)
    x_val, y_val = separable_dataset(10, seed=2)
    data = {"labels": ["a", "b", "c"], "train": {"x": x_train, "y": y_train}, "val": {"x": x_val, "y": y_val}}

    # A small batch on a small set, so 25 epochs are 100 gradient steps rather than 25.
    options = {**temporal.DEFAULTS, "epochs": 25, "batch_size": 16}
    result = temporal.train_temporal(data, options, verbose=False)
    assert result["val_macro_f1"] > 0.6
    assert 0 <= result["best_epoch"] < 25
    assert len(result["history"]) >= 1
    assert result["n_parameters"] > 0


def test_training_is_reproducible_for_a_fixed_seed():
    x, y = separable_dataset(10, seed=1)
    data = {"labels": ["a", "b", "c"], "train": {"x": x, "y": y}, "val": {"x": x, "y": y}}
    options = {**temporal.DEFAULTS, "epochs": 3}
    first = temporal.train_temporal(data, options, seed=7, verbose=False)
    second = temporal.train_temporal(data, options, seed=7, verbose=False)
    assert first["val_macro_f1"] == second["val_macro_f1"]


def test_early_stopping_gives_up_after_patience_epochs():
    """Random labels never improve, so training must stop well before the epoch limit."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (30, feat.WINDOW_FRAMES, feat.N_FEATURES)).astype(np.float32)
    y = rng.integers(0, 3, 30)
    data = {"labels": ["a", "b", "c"], "train": {"x": x, "y": y}, "val": {"x": x, "y": y}}

    result = temporal.train_temporal(data, {**temporal.DEFAULTS, "epochs": 50, "patience": 2}, verbose=False)
    assert len(result["history"]) < 50


def test_saving_a_run_writes_the_checkpoint_and_its_config(tmp_path):
    import json

    x, y = separable_dataset(6, seed=3)
    data = {"labels": ["a", "b", "c"], "train": {"x": x, "y": y}, "val": {"x": x, "y": y}}
    options = {**temporal.DEFAULTS, "epochs": 2}
    result = temporal.train_temporal(data, options, verbose=False)

    out = temporal.save_run(result, data["labels"], tmp_path / "run", options, seed=42)
    assert (out / "temporal.pt").exists()
    metadata = json.loads((out / "temporal.json").read_text())
    assert metadata["labels"] == ["a", "b", "c"]
    assert metadata["options"]["time_scale"] == [0.8, 1.2], "tuples must survive the JSON round-trip"
    assert "model" not in metadata or isinstance(metadata["model"], str)
