"""Export the keypoints of the labelled videos to JSON for the Node rep-counter harness.

Only videos that appear in labels/rep_labels.csv are written: those are the only ones
`node tools/eval/evalReps.js` can score. A frame without a detected pose becomes `null`, which the
harness passes to the counters as "no pose", exactly as the browser does.

    python -m repcount.export.keypoints_json [--labels labels/rep_labels.csv]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from repcount import config
from repcount.features.windows import find_keypoint_files, load_sequence
from repcount.labels.validate import LABELS_PATH

JSON_DIR = config.DATA_DIR / "keypoints_json"
DECIMALS = 5  # well below MediaPipe's own precision, and keeps a clip around 250 KB


# ---------------------------------------------------------------- pure helpers


def frame_to_json(frame: np.ndarray) -> list | None:
    """(33, 4) landmarks → nested lists, or None when MediaPipe found no pose in this frame.

    "No pose" means every landmark is NaN (see repcount.data.extract.pose_to_arrays). A single missing
    value is kept as null in place, so one gap never throws away the other 32 landmarks.
    """
    if not np.isfinite(frame[:, :2]).any():
        return None
    return [[round(float(v), DECIMALS) if math.isfinite(v) else None for v in point] for point in frame]


def video_to_json(sequence: dict) -> dict:
    """One extracted video (repcount.features.windows.load_sequence) → the harness's JSON layout."""
    return {
        "video_id": str(sequence["video_id"]),
        "label": str(sequence["label"]),
        "width": int(sequence["width"]),
        "height": int(sequence["height"]),
        "timestamps_ms": [int(t) for t in sequence["timestamps_ms"]],
        "landmarks": [frame_to_json(frame) for frame in np.asarray(sequence["landmarks"], dtype=np.float64)],
    }


# ---------------------------------------------------------------- I/O


def export(video_ids: list[str], keypoints_dir: Path, out_dir: Path) -> tuple[list[Path], list[str]]:
    """Write one JSON per labelled video; returns the files written and the ids with no keypoints."""
    files = find_keypoint_files(keypoints_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for video_id in video_ids:
        if video_id not in files:
            missing.append(video_id)
            continue
        path = out_dir / f"{video_id}.json"
        path.write_text(json.dumps(video_to_json(load_sequence(files[video_id])), separators=(",", ":")),
                        encoding="utf-8")
        written.append(path)
    return written, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=JSON_DIR)
    args = parser.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"{args.labels} not found — label the videos first (labels/README.md)")
    video_ids = sorted(pd.read_csv(args.labels, dtype=str, keep_default_na=False)["video_id"].unique())
    written, missing = export(video_ids, args.keypoints_dir, args.out_dir)
    size = sum(p.stat().st_size for p in written) / 1e6
    print(f"{len(written)} labelled videos → {args.out_dir} ({size:.1f} MB)")
    for video_id in missing:
        print(f"no keypoints for {video_id}")


if __name__ == "__main__":
    main()
