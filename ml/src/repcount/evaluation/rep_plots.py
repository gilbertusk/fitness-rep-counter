"""Draw rep-counter traces: the counter's signal, the reps it counted and the human labels.

`node tools/eval/evalReps.js` keeps the traces of one clear success and the two worst misses; this
turns them into the figures of reports/03-rep-counter/rep_counter.md.

    python -m repcount.evaluation.rep_plots --traces data/runs/rep_traces_generic_test.json
"""

import argparse
import json
from pathlib import Path

from repcount import config

FIGURES_DIR = config.REPORTS_DIR / "03-rep-counter" / "figures"


def figure_title(trace: dict) -> str:
    truth, counted = len(trace["truth_ms"]), len(trace["predicted_ms"])
    verdict = "tepat" if truth == counted else f"selisih {counted - truth:+d}"
    return f"{trace['video_id']} ({trace['label']}) — label {truth} rep, terhitung {counted} ({verdict})"


def plot_trace(trace: dict, counter: str, path: Path) -> Path:
    """One figure: signal over time, green dashed = human marks, red = counted reps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seconds = [t / 1000 for t in trace["timestamps_ms"]]
    points = [(t, v) for t, v in zip(seconds, trace["signal"], strict=True) if v is not None]
    figure, axes = plt.subplots(figsize=(10, 3.2))
    if points:
        axes.plot(*zip(*points, strict=True), color="#4b5563", linewidth=1.2, label=f"sinyal ({counter})")
    for i, t in enumerate(trace["truth_ms"]):
        axes.axvline(t / 1000, color="#15803d", linestyle="--", linewidth=1.2,
                     label="label manusia" if i == 0 else None)
    for i, t in enumerate(trace["predicted_ms"]):
        axes.axvline(t / 1000, color="#dc2626", linewidth=1.6, alpha=0.8,
                     label="rep terhitung" if i == 0 else None)
    axes.set_xlabel("waktu (detik)")
    axes.set_title(figure_title(trace), fontsize=10)
    axes.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return path


def plot_all(traces_document: dict, out_dir: Path) -> list[Path]:
    counter = traces_document["counter"]
    return [plot_trace(trace, counter, out_dir / f"{counter}_{trace['video_id']}.png")
            for trace in traces_document["traces"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", type=Path, required=True, help="rep_traces_<counter>_<split>.json")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    if not args.traces.exists():
        raise SystemExit(f"{args.traces} not found — run `node tools/eval/evalReps.js` first")
    for path in plot_all(json.loads(args.traces.read_text(encoding="utf-8")), args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
