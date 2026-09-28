"""Temporal exercise classifier: a small 1D-CNN over the 30×47 feature window.

Why a 1D-CNN rather than a GRU:

* the window is a fixed 30 frames, so there is nothing for recurrence to carry across — a stack of
  dilated-free convolutions already sees the whole 2 seconds;
* convolutions export to ONNX as plain Conv nodes, which `onnxruntime-web` runs on both the WASM and
  WebGPU backends; GRU export drags in a loop that the web runtimes handle far less predictably;
* it parallelises over time, so one window costs a single batched pass in the browser instead of 30
  sequential steps.

The whole model is ~63k parameters (about 250 KB as float32), comfortably under the 1 MB budget.

    python -m repcount.models.temporal [--limit N] [--epochs N] [--out DIR]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from repcount import config
from repcount.features import features as feat

DEFAULTS = {
    "epochs": 80,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 12,        # epochs without a better val macro-F1 before stopping
    "flip_probability": 0.5,
    "jitter_std": 0.01,    # in normalized torso units
    "time_scale": (0.8, 1.2),
}


# ---------------------------------------------------------------- augmentation (pure)


def time_scale_window(window: np.ndarray, scale: float) -> np.ndarray:
    """Replay a window as if the movement were `scale`× faster, resampling back to the same length."""
    length = window.shape[0]
    source = np.clip(np.arange(length) * scale, 0, length - 1)
    lower = np.floor(source).astype(int)
    upper = np.minimum(lower + 1, length - 1)
    weight = (source - lower)[:, None]
    return window[lower] * (1 - weight) + window[upper] * weight


def augment_batch(batch: np.ndarray, rng: np.random.Generator, options: dict = DEFAULTS) -> np.ndarray:
    """Flip left/right, scale time and jitter positions — all label-preserving (docs/FEATURES.md §9)."""
    out = np.array(batch, dtype=np.float64, copy=True)
    flip = rng.random(len(out)) < options["flip_probability"]
    out[flip] = feat.flip_features(out[flip])

    low, high = options["time_scale"]
    for i, scale in enumerate(rng.uniform(low, high, len(out))):
        out[i] = time_scale_window(out[i], scale)

    # Jitter positions only: visibility and the angles are not spatial offsets.
    noise = rng.normal(0.0, options["jitter_std"], out.shape)
    columns = np.zeros(feat.N_FEATURES, dtype=bool)
    for point in range(feat.N_POINTS):
        columns[point * 3:point * 3 + 2] = True
    return out + noise * columns


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights; a class absent from the split gets weight 0, not infinity."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(n_classes)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


# ---------------------------------------------------------------- model


def build_model(n_classes: int, n_features: int = feat.N_FEATURES):
    """Conv1d stack over (batch, features, time); returns logits."""
    from torch import nn

    def block(in_channels, out_channels, kernel):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    class TemporalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                block(n_features, 64, 5), block(64, 64, 5), nn.MaxPool1d(2), block(64, 128, 3),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

        def forward(self, windows):  # (batch, time, features) → (batch, n_classes)
            return self.head(self.features(windows.transpose(1, 2)))

    return TemporalClassifier()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- training


def macro_f1(model, x, y, n_classes: int, batch_size: int = 256) -> tuple[float, float]:
    """Validation accuracy and macro-F1, averaged over all `n_classes` even if some are absent."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32)
            predictions.append(model(batch).argmax(dim=1).numpy())
    predicted = np.concatenate(predictions) if predictions else np.empty(0, dtype=int)
    return (
        float(accuracy_score(y, predicted)),
        float(f1_score(y, predicted, average="macro", labels=range(n_classes), zero_division=0)),
    )


def train_temporal(data: dict, options: dict = DEFAULTS, seed: int = config.SEED, verbose: bool = True) -> dict:
    """Train with early stopping on validation macro-F1; the test split is never touched here."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = len(data["labels"])
    x_train, y_train = data["train"]["x"], data["train"]["y"]

    model = build_model(n_classes)
    weights = torch.as_tensor(class_weights(y_train, n_classes), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                  weight_decay=options["weight_decay"])

    best = {"macro_f1": -1.0, "accuracy": 0.0, "epoch": -1, "state": None}
    history = []
    for epoch in range(options["epochs"]):
        model.train()
        order = rng.permutation(len(x_train))
        total_loss = 0.0
        for start in range(0, len(order), options["batch_size"]):
            rows = order[start:start + options["batch_size"]]
            batch = torch.as_tensor(augment_batch(x_train[rows], rng, options), dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(model(batch), torch.as_tensor(y_train[rows], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)

        accuracy, f1 = macro_f1(model, data["val"]["x"], data["val"]["y"], n_classes)
        history.append({"epoch": epoch, "loss": total_loss / max(len(order), 1),
                        "val_accuracy": accuracy, "val_macro_f1": f1})
        if f1 > best["macro_f1"]:
            best = {"macro_f1": f1, "accuracy": accuracy, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if verbose:
            print(f"epoch {epoch:3d} loss {history[-1]['loss']:.4f} val acc {accuracy:.3f} macro-F1 {f1:.3f}"
                  f"{'  ← best' if epoch == best['epoch'] else ''}", flush=True)
        if epoch - best["epoch"] >= options["patience"]:
            break

    model.load_state_dict(best["state"])
    return {"model": model, "history": history, "best_epoch": best["epoch"],
            "val_macro_f1": best["macro_f1"], "val_accuracy": best["accuracy"],
            "n_parameters": count_parameters(model)}


# ---------------------------------------------------------------- I/O


def save_run(result: dict, labels: list[str], out_dir: Path, options: dict, seed: int) -> Path:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), out_dir / "temporal.pt")
    (out_dir / "temporal.json").write_text(json.dumps({
        "model": "1D-CNN over the 30×47 feature window",
        "command": "python -m repcount.models.temporal",
        "seed": seed,
        "labels": labels,
        "options": {k: list(v) if isinstance(v, tuple) else v for k, v in options.items()},
        **{k: v for k, v in result.items() if k != "model"},
    }, indent=2) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    from repcount.features.windows import load_all

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file", type=Path, default=config.SPLITS_DIR / "split_v1.json")
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="use at most N videos per split")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=None, help="run directory (default: data/runs/<timestamp>)")
    args = parser.parse_args()

    started = time.perf_counter()
    data = load_all(args.split_file, args.keypoints_dir, args.limit)
    options = {**DEFAULTS, "epochs": args.epochs}
    print(f"{len(data['train']['x'])} train / {len(data['val']['x'])} val windows, {len(data['labels'])} classes")

    result = train_temporal(data, options, args.seed)
    out_dir = save_run(result, data["labels"], args.out or config.RUNS_DIR / time.strftime("%Y%m%d-%H%M%S"),
                       options, args.seed)
    print(f"best epoch {result['best_epoch']} · val accuracy {result['val_accuracy']:.3f} · "
          f"val macro-F1 {result['val_macro_f1']:.3f} · {result['n_parameters']} parameters "
          f"({time.perf_counter() - started:.1f}s) → {out_dir}")


if __name__ == "__main__":
    main()
