"""Baseline exercise classifier: per-window summary statistics + gradient boosting.

The point of the baseline is to say how much the temporal model in `repcount.models.temporal`
actually buys. It throws away frame order except for one dominant-frequency term per joint angle,
so a temporal model that cannot beat it is not earning its complexity.

    python -m repcount.models.baseline [--limit N] [--out DIR]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from repcount import config
from repcount.features import features as feat

SUMMARY_STATS = ("mean", "std", "min", "max", "range")
N_SUMMARY = len(SUMMARY_STATS) * feat.N_FEATURES + feat.N_ANGLES


# ---------------------------------------------------------------- pure helpers


def dominant_frequency(series: np.ndarray, fps: float = feat.TARGET_FPS) -> float:
    """Frequency (Hz) of the strongest non-constant component of one window's signal.

    A 2-second window at 15 fps resolves 0.5 Hz steps, which is coarse but enough to separate a
    slow deadlift from a fast biceps curl. The DC bin is dropped so a limb that simply sits at a
    constant angle does not report 0 Hz as its "motion".
    """
    values = np.asarray(series, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all() or np.ptp(values) == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(values - values.mean()))
    if spectrum.size < 2:
        return 0.0
    return float(np.fft.rfftfreq(values.size, d=1.0 / fps)[1 + int(np.argmax(spectrum[1:]))])


def summarize_window(window: np.ndarray, fps: float = feat.TARGET_FPS) -> np.ndarray:
    """(WINDOW_FRAMES, N_FEATURES) → one flat summary vector of N_SUMMARY values."""
    values = np.asarray(window, dtype=np.float64)
    stats = np.concatenate([
        values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0), np.ptp(values, axis=0),
    ])
    angles = values[:, feat.N_POINTS * 3:]
    return np.concatenate([stats, [dominant_frequency(angles[:, i], fps) for i in range(feat.N_ANGLES)]])


def summarize(windows: np.ndarray, fps: float = feat.TARGET_FPS) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0, N_SUMMARY))
    return np.stack([summarize_window(window, fps) for window in windows])


def summary_feature_names() -> list[str]:
    """Names in the same order as `summarize_window`, for reading feature importances."""
    columns = [f"p{i}_{axis}" for i in range(feat.N_POINTS) for axis in ("x", "y", "vis")]
    columns += [f"angle{i}" for i in range(feat.N_ANGLES)]
    names = [f"{column}_{stat}" for stat in SUMMARY_STATS for column in columns]
    return names + [f"angle{i}_freq" for i in range(feat.N_ANGLES)]


# ---------------------------------------------------------------- training


def build_model(seed: int = config.SEED):
    """Gradient boosting with balanced class weights — the dataset is heavily imbalanced."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        class_weight="balanced",
        random_state=seed,
    )


def train_baseline(data: dict, seed: int = config.SEED) -> dict:
    """Fit on the train split and score the validation split; the test split is never touched here."""
    from sklearn.metrics import accuracy_score, f1_score

    x_train, x_val = summarize(data["train"]["x"]), summarize(data["val"]["x"])
    model = build_model(seed).fit(x_train, data["train"]["y"])
    predictions = model.predict(x_val)
    # Average over every class, not just the ones present in val, so a class the split leaves out
    # cannot inflate the number (see repcount.evaluation.classifier_report.metrics).
    indices = range(len(data["labels"])) if "labels" in data else range(int(data["train"]["y"].max()) + 1)
    return {
        "model": model,
        "val_accuracy": float(accuracy_score(data["val"]["y"], predictions)),
        "val_macro_f1": float(f1_score(data["val"]["y"], predictions, average="macro",
                                       labels=indices, zero_division=0)),
        "n_train_windows": int(len(x_train)),
        "n_val_windows": int(len(x_val)),
    }


# ---------------------------------------------------------------- I/O


def save_run(result: dict, labels: list[str], out_dir: Path, seed: int) -> Path:
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], out_dir / "baseline.joblib")
    (out_dir / "baseline.json").write_text(json.dumps({
        "model": "HistGradientBoostingClassifier on window summary statistics",
        "command": "python -m repcount.models.baseline",
        "seed": seed,
        "labels": labels,
        "n_summary_features": N_SUMMARY,
        **{k: v for k, v in result.items() if k != "model"},
    }, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    from repcount.features.windows import load_all

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file", type=Path, default=config.SPLITS_DIR / "split_v1.json")
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="use at most N videos per split")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=None, help="run directory (default: data/runs/<timestamp>)")
    args = parser.parse_args()

    started = time.perf_counter()
    data = load_all(args.split_file, args.keypoints_dir, args.limit)
    print(f"{len(data['train']['x'])} train / {len(data['val']['x'])} val windows, {len(data['labels'])} classes")

    result = train_baseline(data, args.seed)
    out_dir = save_run(result, data["labels"], args.out or config.RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"), args.seed)
    print(f"val accuracy {result['val_accuracy']:.3f} · val macro-F1 {result['val_macro_f1']:.3f} "
          f"({time.perf_counter() - started:.1f}s) → {out_dir}")


if __name__ == "__main__":
    main()
