"""Stratified, group-aware train/val/test split by video.

Splits each class 70/15/15 by `group_id` (near-duplicates never cross splits), seed 42.
Only videos with extracted keypoints are split. Classes with < 10 videos still get at least
one video in val and test when possible; exceptions are listed in the report.

    python -m repcount.data.split
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from repcount import config
from repcount.data.extract import output_path

SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SMALL_CLASS = 10
SPLIT_VERSION = "v1"


def split_targets(n_videos: int, ratios: dict = RATIOS) -> dict[str, int]:
    """Target video count per split; val/test get >= 1 each when the class has >= 3 videos."""
    val = round(n_videos * ratios["val"])
    test = round(n_videos * ratios["test"])
    if n_videos >= 3:
        val, test = max(1, val), max(1, test)
    return {"train": n_videos - val - test, "val": val, "test": test}


def assign_class_groups(group_sizes: dict[str, int], targets: dict[str, int], rng: np.random.Generator) -> dict:
    """Assign whole groups to the split with the largest remaining deficit (ties → train, val, test)."""
    names = sorted(group_sizes)
    order = rng.permutation(len(names))
    # Big groups first so they land where there is room; the shuffle breaks ties between equal sizes.
    ranked = sorted((names[i] for i in order), key=lambda g: -group_sizes[g])
    counts = dict.fromkeys(SPLITS, 0)
    assignment = {}
    for group in ranked:
        split = max(SPLITS, key=lambda s: targets[s] - counts[s])
        assignment[group] = split
        counts = {**counts, split: counts[split] + group_sizes[group]}
    return assignment


def group_labels(videos: pd.DataFrame) -> pd.Series:
    """Label per group: the majority label of its videos (groups normally stay within one class)."""
    return videos.groupby("group_id")["label"].agg(lambda s: Counter(s).most_common(1)[0][0])


def make_split(videos: pd.DataFrame, seed: int = config.SEED) -> dict[str, str]:
    """videos with columns video_id, label, group_id → {video_id: split}."""
    rng = np.random.default_rng(seed)
    labels = group_labels(videos)
    sizes = videos["group_id"].value_counts()
    group_split = {}
    for label in sorted(labels.unique()):
        groups = {g: int(sizes[g]) for g in labels[labels == label].index}
        n_videos = sum(groups.values())
        group_split |= assign_class_groups(groups, split_targets(n_videos), rng)
    return {v: group_split[g] for v, g in zip(videos["video_id"], videos["group_id"], strict=True)}


def per_class_counts(videos: pd.DataFrame, assignment: dict[str, str]) -> pd.DataFrame:
    table = videos.assign(split=videos["video_id"].map(assignment))
    counts = table.pivot_table(index="label", columns="split", values="video_id", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=list(SPLITS), fill_value=0)
    return counts.assign(total=counts.sum(axis=1))


def exceptions(counts: pd.DataFrame) -> list[str]:
    notes = []
    for label, row in counts.iterrows():
        if row["total"] < SMALL_CLASS:
            notes.append(f"{label}: hanya {row['total']} video (< {SMALL_CLASS}); "
                         f"train/val/test = {row['train']}/{row['val']}/{row['test']}")
        if row["val"] == 0 or row["test"] == 0:
            notes.append(f"{label}: val atau test kosong")
    return notes


# ---------------------------------------------------------------- I/O


def load_videos(manifest_path: Path, keypoints_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    has_npz = [output_path(keypoints_dir, lbl, vid).exists() for lbl, vid in zip(manifest["label"], manifest["video_id"], strict=True)]
    return manifest[has_npz].reset_index(drop=True), manifest[[not h for h in has_npz]].reset_index(drop=True)


def split_document(videos: pd.DataFrame, excluded: pd.DataFrame, assignment: dict, seed: int) -> dict:
    counts = per_class_counts(videos, assignment)
    return {
        "version": SPLIT_VERSION,
        "seed": seed,
        "ratios": RATIOS,
        "unit": "video, grouped by near-duplicate group_id",
        "command": "python -m repcount.data.split",
        "splits": {s: sorted(v for v, sp in assignment.items() if sp == s) for s in SPLITS},
        "groups": dict(sorted(zip(videos["video_id"], videos["group_id"], strict=True))),
        "per_class": {lbl: {k: int(v) for k, v in row.items()} for lbl, row in counts.iterrows()},
        "excluded": sorted(excluded["video_id"].tolist()),
    }


def render_report(doc: dict, videos: pd.DataFrame, excluded: pd.DataFrame) -> str:
    counts = pd.DataFrame(doc["per_class"]).T
    totals = counts.sum()
    multi = videos["group_id"].value_counts()
    lines = [
        "# Split data v1",
        "",
        f"Perintah: `{doc['command']}` · seed {doc['seed']} · rasio 70/15/15 per kelas, dibagi per `group_id`.",
        "",
        f"- Video dalam split: **{len(videos)}** (dikecualikan: {len(excluded)} — tanpa keypoint: "
        f"{', '.join(excluded['video_id']) or '-'})",
        f"- Grup near-duplicate berisi > 1 video: **{int((multi > 1).sum())}** "
        f"({int(multi[multi > 1].sum())} video); satu grup selalu berada di satu split.",
        f"- Total: train **{totals['train']}** ({totals['train'] / totals['total']:.1%}), "
        f"val **{totals['val']}** ({totals['val'] / totals['total']:.1%}), "
        f"test **{totals['test']}** ({totals['test'] / totals['total']:.1%})",
        "",
        "| Kelas | train | val | test | total |",
        "|---|---:|---:|---:|---:|",
        *[f"| {lbl} | {r['train']} | {r['val']} | {r['test']} | {r['total']} |" for lbl, r in counts.iterrows()],
        "",
        "## Pengecualian",
        "",
        *([f"- {note}" for note in exceptions(counts)] or ["- Tidak ada."]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--out", type=Path, default=config.SPLITS_DIR / f"split_{SPLIT_VERSION}.json")
    parser.add_argument("--report", type=Path, default=config.REPORTS_DIR / "00-data" / "split.md")
    args = parser.parse_args()

    videos, excluded = load_videos(config.MANIFEST_PATH, config.KEYPOINTS_DIR)
    assignment = make_split(videos, args.seed)
    doc = split_document(videos, excluded, assignment, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(doc, videos, excluded), encoding="utf-8")
    print(f"{len(videos)} videos split -> {args.out}, report -> {args.report}")


if __name__ == "__main__":
    main()
