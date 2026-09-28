"""Turn extracted keypoints plus a split file into per-split window datasets.

Windows never cross a video boundary and every window carries its `video_id`, so evaluation can be
aggregated per video (docs/PLAN.md §3) without leaking frames between splits.

    python -m repcount.features.windows [--split-file ...] [--limit N]
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from repcount import config
from repcount.features import features as feat

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- pure helpers


def video_windows(sequence: dict) -> dict:
    """One video's raw keypoints → its usable windows (docs/FEATURES.md §8)."""
    features = feat.sequence_features(
        sequence["landmarks"], sequence["timestamps_ms"], float(sequence["width"]), float(sequence["height"])
    )
    cut = feat.make_windows(features)
    keep = feat.usable(cut["missing_ratio"])
    return {
        "windows": cut["windows"][keep],
        "starts": cut["starts"][keep],
        "missing_ratio": cut["missing_ratio"][keep],
        "n_dropped": int((~keep).sum()),
        "n_frames": features.shape[0],
    }


def stack_dataset(per_video: list[dict], label_to_index: dict[str, int]) -> dict:
    """Concatenate per-video windows into arrays for training."""
    if not per_video:
        empty = np.empty((0, feat.WINDOW_FRAMES, feat.N_FEATURES), dtype=np.float32)
        return {"x": empty, "y": np.empty(0, np.int64), "video_ids": np.empty(0, dtype=object)}
    return {
        "x": np.concatenate([v["windows"] for v in per_video]).astype(np.float32),
        "y": np.concatenate([np.full(len(v["windows"]), label_to_index[v["label"]], np.int64) for v in per_video]),
        "video_ids": np.concatenate([np.full(len(v["windows"]), v["video_id"], dtype=object) for v in per_video]),
    }


# ---------------------------------------------------------------- I/O


def read_split(split_file: Path) -> dict:
    return json.loads(Path(split_file).read_text(encoding="utf-8"))


def load_sequence(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in ("landmarks", "timestamps_ms", "width", "height", "label", "video_id")}


def find_keypoint_files(keypoints_dir: Path) -> dict[str, Path]:
    """video_id → .npz path, scanning the class folders written by repcount.data.extract."""
    return {path.stem: path for path in sorted(Path(keypoints_dir).glob("*/*.npz"))}


def iter_split_videos(split_document: dict, split: str, files: dict[str, Path], limit: int | None) -> Iterator[Path]:
    ids = split_document["splits"].get(split, [])
    found = [files[v] for v in ids if v in files]
    yield from (found[:limit] if limit else found)


def build_split(split_document: dict, split: str, keypoints_dir: Path, label_to_index: dict[str, int],
                limit: int | None = None) -> dict:
    """Read every video of one split and return its stacked window dataset plus counts."""
    files = find_keypoint_files(keypoints_dir)
    per_video, dropped = [], 0
    for path in iter_split_videos(split_document, split, files, limit):
        sequence = load_sequence(path)
        result = video_windows(sequence)
        dropped += result["n_dropped"]
        if len(result["windows"]):
            per_video.append({**result, "label": str(sequence["label"]), "video_id": str(sequence["video_id"])})
    dataset = stack_dataset(per_video, label_to_index)
    return {**dataset, "n_videos": len(per_video), "n_dropped": dropped}


def load_all(split_file: Path | None = None, keypoints_dir: Path | None = None, limit: int | None = None) -> dict:
    """All three splits plus the shared label vocabulary — the entry point used by the model scripts."""
    split_file = split_file or config.SPLITS_DIR / "split_v1.json"
    keypoints_dir = keypoints_dir or config.KEYPOINTS_DIR
    document = read_split(split_file)
    files = find_keypoint_files(keypoints_dir)
    labels = sorted({p.parent.name for p in files.values()})
    label_to_index = {name: i for i, name in enumerate(labels)}
    return {
        "labels": labels,
        **{s: build_split(document, s, keypoints_dir, label_to_index, limit) for s in SPLITS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-file", type=Path, default=config.SPLITS_DIR / "split_v1.json")
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="use at most N videos per split")
    args = parser.parse_args()

    data = load_all(args.split_file, args.keypoints_dir, args.limit)
    print(f"{len(data['labels'])} classes: {', '.join(data['labels'])}")
    for split in SPLITS:
        d = data[split]
        print(f"{split:>5}: {len(d['x']):5d} windows from {d['n_videos']:3d} videos "
              f"({d['n_dropped']} dropped for > {feat.MAX_MISSING_RATIO:.0%} missing frames)")


if __name__ == "__main__":
    main()
