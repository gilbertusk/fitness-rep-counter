"""Unit tests for repcount.models.baseline."""

import numpy as np
import pytest

from repcount.features import features as feat
from repcount.models import baseline


def oscillating_window(frequency_hz: float, n_features: int = feat.N_FEATURES) -> np.ndarray:
    t = np.arange(feat.WINDOW_FRAMES) / feat.TARGET_FPS
    window = np.zeros((feat.WINDOW_FRAMES, n_features))
    window[:, feat.N_POINTS * 3:] = np.sin(2 * np.pi * frequency_hz * t)[:, None]
    return window


def separable_dataset(n_per_class: int, n_classes: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Classes differ only in how fast their joint angles oscillate."""
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    for label in range(n_classes):
        for _ in range(n_per_class):
            noise = rng.normal(0, 0.05, (feat.WINDOW_FRAMES, feat.N_FEATURES))
            windows.append(oscillating_window(0.5 * (label + 1)) + noise)
            labels.append(label)
    return np.array(windows, np.float32), np.array(labels, np.int64)


# ---------------------------------------------------------------- dominant frequency


def test_dominant_frequency_finds_a_pure_oscillation():
    t = np.arange(feat.WINDOW_FRAMES) / feat.TARGET_FPS
    assert baseline.dominant_frequency(np.sin(2 * np.pi * 1.0 * t)) == pytest.approx(1.0, abs=0.3)


def test_dominant_frequency_separates_a_slow_from_a_fast_movement():
    slow = baseline.dominant_frequency(oscillating_window(0.5)[:, -1])
    fast = baseline.dominant_frequency(oscillating_window(2.0)[:, -1])
    assert slow < fast


def test_dominant_frequency_of_a_constant_signal_is_zero():
    assert baseline.dominant_frequency(np.full(feat.WINDOW_FRAMES, 0.7)) == 0.0


def test_dominant_frequency_ignores_the_dc_component():
    """A large offset must not make the strongest bin 0 Hz."""
    t = np.arange(feat.WINDOW_FRAMES) / feat.TARGET_FPS
    assert baseline.dominant_frequency(100 + np.sin(2 * np.pi * 1.0 * t)) == pytest.approx(1.0, abs=0.3)


def test_dominant_frequency_is_zero_for_unusable_input():
    assert baseline.dominant_frequency(np.array([1.0])) == 0.0
    assert baseline.dominant_frequency(np.array([np.nan] * feat.WINDOW_FRAMES)) == 0.0
    assert baseline.dominant_frequency(np.array([])) == 0.0


# ---------------------------------------------------------------- summaries


def test_summarize_window_has_the_documented_length():
    summary = baseline.summarize_window(oscillating_window(1.0))
    assert summary.shape == (baseline.N_SUMMARY,)
    assert len(baseline.SUMMARY_STATS) * feat.N_FEATURES + feat.N_ANGLES == baseline.N_SUMMARY


def test_summarize_window_computes_the_statistics_it_claims():
    window = np.arange(feat.WINDOW_FRAMES * feat.N_FEATURES, dtype=float)
    window = window.reshape(feat.WINDOW_FRAMES, feat.N_FEATURES)
    summary = baseline.summarize_window(window)
    assert summary[:feat.N_FEATURES] == pytest.approx(window.mean(axis=0))
    assert summary[2 * feat.N_FEATURES:3 * feat.N_FEATURES] == pytest.approx(window.min(axis=0))
    assert summary[3 * feat.N_FEATURES:4 * feat.N_FEATURES] == pytest.approx(window.max(axis=0))


def test_summarize_stacks_every_window():
    windows = np.stack([oscillating_window(1.0), oscillating_window(2.0)])
    assert baseline.summarize(windows).shape == (2, baseline.N_SUMMARY)


def test_summarize_handles_an_empty_split():
    assert baseline.summarize(np.empty((0, feat.WINDOW_FRAMES, feat.N_FEATURES))).shape == (0, baseline.N_SUMMARY)


def test_feature_names_line_up_with_the_summary_vector():
    names = baseline.summary_feature_names()
    assert len(names) == baseline.N_SUMMARY
    assert len(set(names)) == baseline.N_SUMMARY, "feature names must be unique to be readable"
    assert names[0] == "p0_x_mean"
    assert names[-1] == f"angle{feat.N_ANGLES - 1}_freq"


# ---------------------------------------------------------------- training


def test_the_baseline_learns_a_separable_dataset():
    x_train, y_train = separable_dataset(40, seed=1)
    x_val, y_val = separable_dataset(15, seed=2)
    data = {"train": {"x": x_train, "y": y_train}, "val": {"x": x_val, "y": y_val}}

    result = baseline.train_baseline(data)
    assert result["val_macro_f1"] > 0.8
    assert result["n_train_windows"] == len(x_train)
    assert result["n_val_windows"] == len(x_val)


def test_the_baseline_balances_class_weights():
    assert baseline.build_model().class_weight == "balanced"


def test_saving_a_run_writes_the_model_and_its_metadata(tmp_path):
    x_train, y_train = separable_dataset(20, n_classes=2, seed=3)
    data = {"train": {"x": x_train, "y": y_train}, "val": {"x": x_train, "y": y_train}}
    result = baseline.train_baseline(data)

    out = baseline.save_run(result, ["a", "b"], tmp_path / "run", seed=42)
    assert (out / "baseline.joblib").exists()

    import json
    metadata = json.loads((out / "baseline.json").read_text())
    assert metadata["labels"] == ["a", "b"]
    assert metadata["seed"] == 42
    assert "val_macro_f1" in metadata
    assert "model" not in metadata or isinstance(metadata["model"], str), "the fitted estimator must not be inlined"
