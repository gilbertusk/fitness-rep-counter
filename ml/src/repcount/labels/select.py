"""Choose the videos a human labels for repetitions → labels/to_label.csv.

Only the val and test splits are used: the labels are evaluation ground truth for stage 3, and a rep
counter must never be tuned on the videos it is scored on. At most `--per-class` videos per class,
picked greedily for variety — a new near-duplicate group first (two clips from the same source are the
same person in the same room), then a new file extension (phone `.mov` vs. web `.mp4`), then the
duration farthest from what is already picked. The seed only breaks ties.

Videos where MediaPipe found no pose in any frame are left out: no counter can be evaluated on them.

    python -m repcount.labels.select [--per-class 5] [--seed 42]
"""

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from repcount import config

LABELS_DIR = config.REPO_ROOT / "labels"
TO_LABEL_PATH = LABELS_DIR / "to_label.csv"

PER_CLASS = 5
SOURCE_SPLITS = ("val", "test")
HOLD_CLASSES = frozenset({"plank"})  # isometric: labelled as a hold, not as repetitions

TO_LABEL_COLUMNS = [
    "video_id", "label", "task", "path", "ext", "fps", "duration_s", "split", "group_id", "detected_pct",
]


# ---------------------------------------------------------------- pure helpers


def task_for(label: str) -> str:
    return "hold" if label in HOLD_CLASSES else "reps"


def candidates(manifest: pd.DataFrame, split_document: dict) -> pd.DataFrame:
    """Manifest rows of the val/test videos that opened cleanly, with their split attached."""
    split_of = {v: s for s in SOURCE_SPLITS for v in split_document["splits"].get(s, [])}
    usable = manifest[manifest["video_id"].isin(split_of) & (manifest["error"].fillna("") == "")]
    return usable.assign(split=usable["video_id"].map(split_of)).reset_index(drop=True)


def _class_rng(label: str, seed: int) -> np.random.Generator:
    """One generator per class, so adding or removing a class never reshuffles the others."""
    return np.random.default_rng([seed, zlib.crc32(label.encode("utf-8"))])


def _log_durations(rows: pd.DataFrame) -> np.ndarray:
    durations = pd.to_numeric(rows["duration_s"], errors="coerce")
    durations = durations.fillna(durations.median() if durations.notna().any() else 1.0)
    return np.log(np.clip(durations.to_numpy(dtype=float), 0.1, None))


def pick_diverse(rows: pd.DataFrame, k: int, seed: int = config.SEED) -> list[str]:
    """Greedy pick of `k` video_ids from one class, most varied first (see module docstring)."""
    if rows.empty or k <= 0:
        return []
    order = _class_rng(str(rows["label"].iloc[0]), seed).permutation(len(rows))
    rows = rows.iloc[order].reset_index(drop=True)
    durations = _log_durations(rows)

    first = int(np.argmin(np.abs(durations - np.median(durations))))  # start from a typical clip
    chosen = [first]
    while len(chosen) < min(k, len(rows)):
        groups = set(rows.loc[chosen, "group_id"])
        extensions = set(rows.loc[chosen, "ext"])
        best, best_key = None, None
        for i in range(len(rows)):
            if i in chosen:
                continue
            key = (
                rows.at[i, "group_id"] not in groups,
                rows.at[i, "ext"] not in extensions,
                min(abs(durations[i] - durations[j]) for j in chosen),
            )
            if best_key is None or key > best_key:  # strict: ties keep the earlier (shuffled) row
                best, best_key = i, key
        chosen.append(best)
    return rows.loc[chosen, "video_id"].tolist()


def select_videos(pool: pd.DataFrame, per_class: int = PER_CLASS, seed: int = config.SEED) -> pd.DataFrame:
    """Up to `per_class` videos per class, as rows in the to_label.csv layout."""
    picked = [vid for _, rows in pool.groupby("label", sort=True) for vid in pick_diverse(rows, per_class, seed)]
    chosen = pool[pool["video_id"].isin(picked)].copy()
    chosen["task"] = chosen["label"].map(task_for)
    if "detected_pct" not in chosen:
        chosen["detected_pct"] = math.nan
    return chosen.sort_values(["label", "video_id"])[TO_LABEL_COLUMNS].reset_index(drop=True)


def drop_poseless(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off videos whose detection rate is known to be exactly 0; unknown rates are kept."""
    poseless = pool["detected_pct"].fillna(-1) == 0
    return pool[~poseless].reset_index(drop=True), pool[poseless].reset_index(drop=True)


def shortfalls(pool: pd.DataFrame, chosen: pd.DataFrame, per_class: int = PER_CLASS) -> dict[str, int]:
    """Classes that could not supply `per_class` videos → how many they did supply."""
    counts = chosen["label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in sorted(pool["label"].unique())
            if counts.get(label, 0) < per_class}


# ---------------------------------------------------------------- I/O


def detection_rate(npz_path: Path) -> float:
    """Percentage of frames in which MediaPipe found a pose."""
    with np.load(npz_path, allow_pickle=False) as data:
        landmarks = data["landmarks"]
    if len(landmarks) == 0:
        return 0.0
    return float(100.0 * np.isfinite(landmarks[:, :, 0]).all(axis=1).mean())


def attach_detection_rates(pool: pd.DataFrame, keypoints_dir: Path) -> pd.DataFrame:
    """detected_pct per video from its .npz; NaN when the keypoints are not on this machine."""
    from repcount.data.extract import output_path

    rates = []
    for label, video_id in zip(pool["label"], pool["video_id"], strict=True):
        path = output_path(keypoints_dir, label, video_id)
        rates.append(round(detection_rate(path), 1) if path.exists() else math.nan)
    return pool.assign(detected_pct=rates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--split-file", type=Path, default=config.SPLITS_DIR / "split_v1.json")
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=TO_LABEL_PATH)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"{args.manifest} not found — run stage 0 first (`python -m repcount.data.manifest`)")
    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""])
    split_document = json.loads(args.split_file.read_text(encoding="utf-8"))

    pool, poseless = drop_poseless(attach_detection_rates(candidates(manifest, split_document), args.keypoints_dir))
    chosen = select_videos(pool, args.per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.out, index=False)

    print(f"{len(chosen)} videos from {chosen['label'].nunique()} classes "
          f"({', '.join(SOURCE_SPLITS)} only) → {args.out}")
    if poseless.empty and pool["detected_pct"].isna().all():
        print("note: no keypoints found, so videos without any detected pose could not be excluded")
    for video_id in poseless["video_id"]:
        print(f"left out (no pose in any frame): {video_id}")
    for label, n in shortfalls(pool, chosen, args.per_class).items():
        print(f"only {n} available for {label}")


if __name__ == "__main__":
    main()
