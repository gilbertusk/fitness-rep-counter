"""Export the trained temporal classifier to ONNX for the browser.

Writes `app/models/exercise_classifier.onnx` and `app/models/labels.json`, then checks the exported
graph against PyTorch on 100 random windows. The browser must see the same numbers the evaluation
report was written from, so a drift above the tolerance fails the export instead of shipping.

    python -m repcount.export.onnx --run data/runs/<timestamp>
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from repcount import config
from repcount.features import features as feat

MODELS_DIR = config.REPO_ROOT / "app" / "models"
MODEL_PATH = MODELS_DIR / "exercise_classifier.onnx"
LABELS_PATH = MODELS_DIR / "labels.json"

# 18 is what torch's exporter emits for this graph; asking for less makes it attempt a
# down-conversion that fails on the AveragePool axes and silently leaves the model at 18 anyway.
OPSET = 18
TOLERANCE = 1e-4
N_VERIFY = 100
INPUT_NAME = "window"
OUTPUT_NAME = "scores"


# ---------------------------------------------------------------- export


def export_model(model, path: Path, n_features: int = feat.N_FEATURES, opset: int = OPSET) -> Path:
    """Write the ONNX graph with a dynamic batch axis so the browser can score one window at a time.

    `external_data=False` matters: the exporter otherwise spills the weights into a sibling
    `<name>.onnx.data`, and `app/models/` would ship a 24 KB graph with nothing in it.
    """
    import torch

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, feat.WINDOW_FRAMES, n_features, dtype=torch.float32)
    torch.onnx.export(
        model, (example,), str(path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_shapes={"windows": {0: torch.export.Dim("batch")}},
        opset_version=opset,
        external_data=False,
    )
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        raise SystemExit(f"weights landed in {sidecar} instead of the model file — refusing to ship a stub")
    return path


def verify_export(model, path: Path, n_windows: int = N_VERIFY, seed: int = config.SEED) -> dict:
    """Largest absolute difference between PyTorch and onnxruntime on random windows."""
    import onnxruntime
    import torch

    rng = np.random.default_rng(seed)
    windows = rng.normal(0.0, 1.0, (n_windows, feat.WINDOW_FRAMES, feat.N_FEATURES)).astype(np.float32)

    model.eval()
    with torch.no_grad():
        expected = model(torch.as_tensor(windows)).numpy()

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: windows})[0]
    return {
        "n_windows": n_windows,
        "max_abs_difference": float(np.abs(expected - actual).max()),
        "tolerance": TOLERANCE,
    }


def measure_latency(path: Path, n_runs: int = 200, seed: int = config.SEED) -> dict:
    """Median and p95 CPU time for a single window — the number the report quotes."""
    import onnxruntime

    rng = np.random.default_rng(seed)
    window = rng.normal(0.0, 1.0, (1, feat.WINDOW_FRAMES, feat.N_FEATURES)).astype(np.float32)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for _ in range(20):  # warm up, so the first allocations do not land in the measurement
        session.run([OUTPUT_NAME], {INPUT_NAME: window})
    timings = []
    for _ in range(n_runs):
        started = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: window})
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "n_runs": n_runs,
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


# ---------------------------------------------------------------- metadata


def labels_document(labels: list[str], thresholds: dict | None, verification: dict, latency: dict,
                    model_bytes: int) -> dict:
    """What `app/src/adapters/onnxClassifier.js` fetches next to the model."""
    return {
        "labels": labels,
        # Empty until the validation sweep in classifier_report has run; classifier.js then keeps
        # its own provisional defaults rather than pretending these were measured.
        "thresholds": thresholds or {},
        "window": {
            "frames": feat.WINDOW_FRAMES,
            "stride": feat.WINDOW_STRIDE,
            "features": feat.N_FEATURES,
            "targetFps": feat.TARGET_FPS,
            "spec": "docs/FEATURES.md",
        },
        "export": {
            "command": "python -m repcount.export.onnx",
            "opset": OPSET,
            "modelBytes": model_bytes,
            "verification": verification,
            "latencyCpu": latency,
        },
    }


def load_temporal(run_dir: Path, labels: list[str]):
    import torch

    from repcount.models.temporal import build_model

    model = build_model(len(labels))
    model.load_state_dict(torch.load(run_dir / "temporal.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model


def read_run_metadata(run_dir: Path) -> dict:
    path = run_dir / "temporal.json"
    if not path.exists():
        raise SystemExit(f"{path} not found — train a model first with `python -m repcount.models.temporal`")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory holding temporal.pt")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--labels-out", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    metadata = read_run_metadata(args.run)
    labels = metadata["labels"]
    model = load_temporal(args.run, labels)

    export_model(model, args.model_out)
    verification = verify_export(model, args.model_out)
    if verification["max_abs_difference"] > TOLERANCE:
        raise SystemExit(
            f"ONNX output drifts from PyTorch by {verification['max_abs_difference']:.2e} "
            f"(tolerance {TOLERANCE:.0e}) — not exporting a model the browser would disagree with"
        )
    latency = measure_latency(args.model_out)

    thresholds = json.loads((args.run / "thresholds.json").read_text(encoding="utf-8")) \
        if (args.run / "thresholds.json").exists() else None
    document = labels_document(labels, thresholds, verification, latency, args.model_out.stat().st_size)
    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    args.labels_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    print(f"{args.model_out} ({args.model_out.stat().st_size / 1024:.0f} KB, opset {OPSET}) · "
          f"max |onnx - torch| = {verification['max_abs_difference']:.2e} over {N_VERIFY} windows · "
          f"{latency['median_ms']:.2f} ms/window (median, CPU) → {args.labels_out}")


if __name__ == "__main__":
    main()
