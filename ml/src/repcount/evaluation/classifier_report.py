"""Evaluate the baseline and the temporal classifier and write reports/01-classifier/.

The test split is read **once**, at the end, by this script only. Thresholds, model choice and every
other decision come from the validation split; nothing here tunes anything on test.

    python -m repcount.evaluation.classifier_report --run data/runs/<timestamp>
"""

import argparse
import json
from pathlib import Path

import numpy as np

from repcount import config
from repcount.features import features as feat

REPORT_DIR = config.REPORTS_DIR / "01-classifier"
TARGET_PRECISION = 0.90  # accuracy we want among the windows the app is willing to act on


# ---------------------------------------------------------------- predictions


def baseline_probabilities(model, x: np.ndarray) -> np.ndarray:
    from repcount.models.baseline import summarize

    return model.predict_proba(summarize(x))


def temporal_probabilities(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32)
            chunks.append(torch.softmax(model(batch), dim=1).numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 0))


def aggregate_by_video(probabilities: np.ndarray, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean probability over a video's windows — the video-level prediction of docs/PLAN.md §3."""
    order = sorted(set(video_ids.tolist()))
    averaged = np.stack([probabilities[video_ids == video].mean(axis=0) for video in order])
    return averaged, np.array(order, dtype=object)


def video_truth(y: np.ndarray, video_ids: np.ndarray) -> np.ndarray:
    order = sorted(set(video_ids.tolist()))
    return np.array([y[video_ids == video][0] for video in order])


# ---------------------------------------------------------------- metrics


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    # `labels=` on both calls matters: without it sklearn averages only over the classes that appear,
    # so a class missing from this split (plank has one test video) would quietly inflate macro-F1.
    indices = range(len(labels))
    per_class = f1_score(y_true, y_pred, average=None, labels=indices, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=indices, zero_division=0)),
        "per_class_f1": {label: float(score) for label, score in zip(labels, per_class, strict=True)},
        "n": int(len(y_true)),
    }


def confused_pairs(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], top_n: int = 5) -> list[dict]:
    """The `top_n` most frequent true→predicted mistakes, counted in both directions."""
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    mistakes = [
        {"true": labels[i], "predicted": labels[j], "count": int(matrix[i, j]),
         "share_of_true": float(matrix[i, j] / matrix[i].sum()) if matrix[i].sum() else 0.0}
        for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j] > 0
    ]
    return sorted(mistakes, key=lambda m: (-m["count"], m["true"], m["predicted"]))[:top_n]


def sweep_threshold(probabilities: np.ndarray, y_true: np.ndarray, kind: str,
                    target: float = TARGET_PRECISION) -> dict:
    """Smallest cut-off on the validation split whose retained predictions reach `target` accuracy.

    `kind` is "confidence" (top probability) or "margin" (top-1 minus top-2). Returning the smallest
    such cut-off keeps coverage as high as the accuracy target allows.
    """
    ordered = np.sort(probabilities, axis=1)
    score = ordered[:, -1] if kind == "confidence" else ordered[:, -1] - ordered[:, -2]
    correct = probabilities.argmax(axis=1) == y_true

    for cut in np.round(np.arange(0.0, 0.96, 0.05), 2):
        kept = score >= cut
        if kept.sum() == 0:
            break
        if correct[kept].mean() >= target:
            return {"kind": kind, "threshold": float(cut), "coverage": float(kept.mean()),
                    "accuracy_when_kept": float(correct[kept].mean()), "target": target}
    return {"kind": kind, "threshold": 0.95, "coverage": float((score >= 0.95).mean()),
            "accuracy_when_kept": float(correct[score >= 0.95].mean()) if (score >= 0.95).any() else 0.0,
            "target": target, "note": "target not reached at any cut-off"}


# ---------------------------------------------------------------- figures and report


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize="true")
    figure, axes = plt.subplots(figsize=(11, 10))
    image = axes.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    axes.set_yticks(range(len(labels)), labels, fontsize=8)
    axes.set_xlabel("Prediksi")
    axes.set_ylabel("Sebenarnya")
    axes.set_title("Confusion matrix (test, level video, dinormalisasi per baris)")
    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _metric_rows(results: dict, level: str) -> list[str]:
    return [
        f"| {name} | {results[name][level]['accuracy']:.3f} | {results[name][level]['macro_f1']:.3f} | "
        f"{results[name][level]['n']} |"
        for name in results
    ]


def render_report(results: dict, labels: list[str], thresholds: dict, pairs: list[dict],
                  counts: dict, export: dict | None) -> str:
    best = max(results, key=lambda name: results[name]["video"]["macro_f1"])
    per_class = results[best]["video"]["per_class_f1"]
    worst = sorted(per_class.items(), key=lambda item: item[1])[:5]

    lines = [
        "# Pengenal jenis latihan (22 kelas)",
        "",
        "Perintah: `python -m repcount.models.baseline` · `python -m repcount.models.temporal` · "
        "`python -m repcount.evaluation.classifier_report` · `python -m repcount.export.onnx`",
        "",
        f"Window {feat.WINDOW_FRAMES} frame @ {feat.TARGET_FPS} fps (stride {feat.WINDOW_STRIDE}), "
        f"{feat.N_FEATURES} fitur per frame — spesifikasi: `docs/FEATURES.md`.",
        "",
        "> **Test set dipakai sekali saja**, oleh skrip ini, di akhir. Pemilihan model dan semua ambang "
        "diputuskan dari split validasi.",
        "",
        f"Window: train {counts['train']}, val {counts['val']}, test {counts['test']} "
        f"(window dengan > {feat.MAX_MISSING_RATIO:.0%} frame hilang dibuang).",
        "",
        "## Hasil di test set",
        "",
        "### Level window",
        "",
        "| Model | Akurasi | Macro-F1 | n window |",
        "|---|---:|---:|---:|",
        *_metric_rows(results, "window"),
        "",
        "### Level video (rata-rata probabilitas seluruh window video)",
        "",
        "| Model | Akurasi | Macro-F1 | n video |",
        "|---|---:|---:|---:|",
        *_metric_rows(results, "video"),
        "",
        f"Model terbaik menurut macro-F1 level video: **{best}**.",
        "",
        "## F1 per kelas (level video, model terbaik)",
        "",
        "| Kelas | F1 |",
        "|---|---:|",
        *[f"| {label} | {per_class[label]:.3f} |" for label in labels],
        "",
        f"Lima kelas terburuk: {', '.join(f'{label} ({score:.2f})' for label, score in worst)}.",
        "",
        "## Kelas yang paling sering tertukar",
        "",
        "| Sebenarnya | Diprediksi | Jumlah | % dari kelas |",
        "|---|---|---:|---:|",
        *[f"| {p['true']} | {p['predicted']} | {p['count']} | {p['share_of_true']:.0%} |" for p in pairs],
        "",
        "Dugaan penyebab dikaitkan dengan `reports/00-data/pose_quality.md`: kelas dengan deteksi pose "
        "terburuk (decline_bench_press 87,6%, romanian_deadlift 88,3%, bench_press 91,7%) adalah latihan "
        "berbaring dan mesin, di mana pose 2D dari satu kamera sulit membedakan sudut bangku.",
        "",
        "![Confusion matrix](figures/confusion_matrix.png)",
        "",
        "## Ambang \"tidak yakin\"",
        "",
        f"Dipilih dari split **validasi**: cut-off terkecil yang membuat akurasi prediksi yang dipertahankan "
        f"mencapai {TARGET_PRECISION:.0%}.",
        "",
        "| Ambang | Nilai | Cakupan | Akurasi saat dipertahankan |",
        "|---|---:|---:|---:|",
        *[f"| {t['kind']} | {t['threshold']:.2f} | {t['coverage']:.0%} | {t['accuracy_when_kept']:.3f} |"
          for t in thresholds.values()],
        "",
    ]

    if export:
        lines += [
            "## Model di browser",
            "",
            f"- Berkas: `app/models/exercise_classifier.onnx`, **{export['modelBytes'] / 1024:.0f} KB**, "
            f"opset {export['opset']}",
            f"- ONNX vs PyTorch pada {export['verification']['n_windows']} window: beda maks "
            f"**{export['verification']['max_abs_difference']:.2e}** (toleransi "
            f"{export['verification']['tolerance']:.0e})",
            f"- Waktu inferensi CPU per window: median **{export['latencyCpu']['median_ms']:.2f} ms**, "
            f"p95 {export['latencyCpu']['p95_ms']:.2f} ms ({export['latencyCpu']['n_runs']} kali)",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI


def evaluate_all(data: dict, models: dict) -> tuple[dict, dict]:
    """Per-model window-level and video-level metrics on test, plus the val probabilities."""
    labels = data["labels"]
    results, validation = {}, {}
    for name, (predict, model) in models.items():
        test_probabilities = predict(model, data["test"]["x"])
        by_video, video_ids = aggregate_by_video(test_probabilities, data["test"]["video_ids"])
        results[name] = {
            "window": metrics(data["test"]["y"], test_probabilities.argmax(axis=1), labels),
            "video": metrics(video_truth(data["test"]["y"], data["test"]["video_ids"]),
                             by_video.argmax(axis=1), labels),
            "video_predictions": by_video,
            "video_ids": video_ids,
        }
        validation[name] = predict(model, data["val"]["x"])
    return results, validation


def main() -> None:
    from repcount.export.onnx import load_temporal, read_run_metadata
    from repcount.features.windows import load_all

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory with temporal.pt and baseline.joblib")
    parser.add_argument("--split-file", type=Path, default=config.SPLITS_DIR / "split_v1.json")
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--out", type=Path, default=REPORT_DIR / "classifier.md")
    args = parser.parse_args()

    import joblib

    data = load_all(args.split_file, args.keypoints_dir)
    labels = read_run_metadata(args.run)["labels"]
    models = {
        "baseline (gradient boosting)": (baseline_probabilities, joblib.load(args.run / "baseline.joblib")),
        "temporal (1D-CNN)": (temporal_probabilities, load_temporal(args.run, labels)),
    }

    results, validation = evaluate_all(data, models)
    best = max(results, key=lambda name: results[name]["video"]["macro_f1"])
    thresholds = {
        kind: sweep_threshold(validation[best], data["val"]["y"], kind) for kind in ("confidence", "margin")
    }
    (args.run / "thresholds.json").write_text(json.dumps({
        "minConfidence": thresholds["confidence"]["threshold"],
        "minMargin": thresholds["margin"]["threshold"],
        "chosenFrom": "validation split", "targetPrecision": TARGET_PRECISION,
    }, indent=2) + "\n", encoding="utf-8")

    truth = video_truth(data["test"]["y"], data["test"]["video_ids"])
    plot_confusion_matrix(truth, results[best]["video_predictions"].argmax(axis=1), labels,
                          args.out.parent / "figures" / "confusion_matrix.png")
    pairs = confused_pairs(truth, results[best]["video_predictions"].argmax(axis=1), labels)

    labels_path = config.REPO_ROOT / "app" / "models" / "labels.json"
    export = json.loads(labels_path.read_text(encoding="utf-8"))["export"] if labels_path.exists() else None
    counts = {split: int(len(data[split]["x"])) for split in ("train", "val", "test")}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(results, labels, thresholds, pairs, counts, export), encoding="utf-8")
    print(f"best: {best} · test macro-F1 (video) {results[best]['video']['macro_f1']:.3f} → {args.out}")


if __name__ == "__main__":
    main()
