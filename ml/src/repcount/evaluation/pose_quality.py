"""Pose quality report per class from the extracted keypoints.

    python -m repcount.evaluation.pose_quality
→ reports/00-data/pose_quality.md + pose_quality.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from repcount import config
from repcount.data.extract import output_path

# MediaPipe Pose landmark indices (left, right).
JOINT_GROUPS = {
    "shoulder": (11, 12),
    "elbow": (13, 14),
    "wrist": (15, 16),
    "hip": (23, 24),
    "knee": (25, 26),
    "ankle": (27, 28),
}
JOINT_NAMES_ID = {
    "shoulder": "bahu",
    "elbow": "siku",
    "wrist": "pergelangan tangan",
    "hip": "pinggul",
    "knee": "lutut",
    "ankle": "pergelangan kaki",
}
LEFT = [left for left, _ in JOINT_GROUPS.values()]
RIGHT = [right for _, right in JOINT_GROUPS.values()]
SIDE_MARGIN = 0.05  # mean visibility difference below this counts as "seimbang"
SIDE_NAMES_ID = {"left": "kiri", "right": "kanan", "balanced": "seimbang"}
LOW_DETECTION = 0.80  # class-level warning
LOW_VIDEO_DETECTION = 0.50  # video-level warning


def dominant_side(left_vis: float, right_vis: float, margin: float = SIDE_MARGIN) -> str:
    if np.isnan(left_vis) or np.isnan(right_vis):
        return "unknown"
    if abs(left_vis - right_vis) < margin:
        return "balanced"
    return "left" if left_vis > right_vis else "right"


def video_quality(landmarks: np.ndarray) -> dict:
    """(T, 33, 4) landmarks with NaN rows for missing poses → per-video quality stats."""
    visibility = landmarks[:, :, 3]
    detected = ~np.isnan(visibility).all(axis=1)
    vis = visibility[detected]
    has_pose = bool(detected.any())

    def mean_vis(indices: list[int]) -> float:
        return float(vis[:, indices].mean()) if has_pose else float("nan")

    stats = {"n_frames": int(len(landmarks)), "n_detected": int(detected.sum())}
    stats |= {f"vis_{name}": mean_vis(list(pair)) for name, pair in JOINT_GROUPS.items()}
    stats["side"] = dominant_side(mean_vis(LEFT), mean_vis(RIGHT))
    return stats


def side_summary(sides: pd.Series) -> str:
    counts = sides[sides != "unknown"].value_counts()
    if counts.empty:
        return "-"
    return f"{SIDE_NAMES_ID[counts.index[0]]} ({counts.iloc[0] / counts.sum():.0%})"


def aggregate_by_class(videos: pd.DataFrame) -> pd.DataFrame:
    """Per-video rows (label, duration_s, n_frames, n_detected, vis_*, side) → per-class table, worst first.

    Visibility is weighted by detected frames so long videos count proportionally.
    """
    vis_cols = [f"vis_{name}" for name in JOINT_GROUPS]
    weighted = videos[vis_cols].mul(videos["n_detected"], axis=0).fillna(0.0)
    grouped = videos.assign(**{c: weighted[c] for c in vis_cols}).groupby("label")
    table = grouped.agg(
        n_videos=("label", "size"),
        duration_s=("duration_s", "sum"),
        n_frames=("n_frames", "sum"),
        n_detected=("n_detected", "sum"),
        **{c: (c, "sum") for c in vis_cols},
    )
    table[vis_cols] = table[vis_cols].div(table["n_detected"].replace(0, np.nan), axis=0)
    table["detected_pct"] = 100 * table["n_detected"] / table["n_frames"]
    table["side"] = grouped["side"].agg(side_summary)
    return table.sort_values("detected_pct").reset_index()


def findings(table: pd.DataFrame, videos: pd.DataFrame) -> list[str]:
    """3-5 short findings (Indonesian), derived only from the numbers in `table`."""
    worst, best = table.iloc[0], table.iloc[-1]
    overall = 100 * table["n_detected"].sum() / table["n_frames"].sum()
    low = table[table["detected_pct"] < 100 * LOW_DETECTION]["label"].tolist()
    vis_cols = [f"vis_{name}" for name in JOINT_GROUPS]
    joint_means = table[vis_cols].mean()
    weakest = joint_means.idxmin()
    weak_classes = table.nsmallest(3, weakest)["label"].tolist()
    no_pose = videos[videos["n_detected"] == 0]["video_id"].tolist()
    weak_videos = videos[videos["n_detected"] < LOW_VIDEO_DETECTION * videos["n_frames"]]
    side_counts = videos["side"].value_counts(normalize=True)
    return [
        f"Pose terdeteksi pada {overall:.1f}% dari seluruh frame. Terburuk: **{worst['label']}** "
        f"({worst['detected_pct']:.1f}%), terbaik: **{best['label']}** ({best['detected_pct']:.1f}%).",
        f"Kelas dengan deteksi < {LOW_DETECTION:.0%}: {', '.join(low) if low else 'tidak ada'}; "
        f"namun {len(weak_videos)} video punya deteksi < {LOW_VIDEO_DETECTION:.0%} "
        f"({', '.join(f'{k} {v}' for k, v in weak_videos['label'].value_counts().items()) or '-'}).",
        f"Kelompok sendi dengan visibility rata-rata terendah: **{JOINT_NAMES_ID[weakest.removeprefix('vis_')]}** "
        f"({joint_means[weakest]:.2f}); paling rendah pada {', '.join(weak_classes)}.",
        f"Sisi tubuh dominan per video: kiri {side_counts.get('left', 0):.0%}, kanan "
        f"{side_counts.get('right', 0):.0%}, seimbang {side_counts.get('balanced', 0):.0%} "
        "(pemilihan sisi otomatis di app tetap diperlukan).",
        f"Video tanpa satu pun pose terdeteksi: {len(no_pose)}"
        + (f" ({', '.join(no_pose[:8])}{', …' if len(no_pose) > 8 else ''})." if no_pose else "."),
    ]


# ---------------------------------------------------------------- I/O


def load_video_stats(manifest: pd.DataFrame, keypoints_dir: Path) -> pd.DataFrame:
    rows = []
    for rec in manifest.to_dict("records"):
        path = output_path(keypoints_dir, rec["label"], rec["video_id"])
        if not path.exists():
            continue
        with np.load(path) as data:
            stats = video_quality(data["landmarks"])
            duration = stats["n_frames"] / float(data["fps_effective"])
        rows.append({"video_id": rec["video_id"], "label": rec["label"], "duration_s": duration, **stats})
    return pd.DataFrame(rows)


def render_markdown(table: pd.DataFrame, videos: pd.DataFrame, n_missing: int) -> str:
    joints = " | ".join(JOINT_NAMES_ID[n] for n in JOINT_GROUPS)
    header = f"| Kelas | Video | Durasi (menit) | Pose terdeteksi | {joints} | Sisi dominan |"
    rows = [
        f"| {r['label']} | {r['n_videos']} | {r['duration_s'] / 60:.1f} | {r['detected_pct']:.1f}% | "
        + " | ".join(f"{r[f'vis_{n}']:.2f}" for n in JOINT_GROUPS)
        + f" | {r['side']} |"
        for _, r in table.iterrows()
    ]
    return "\n".join(
        [
            "# Kualitas pose per kelas",
            "",
            "Perintah: `python -m repcount.evaluation.pose_quality` · model `pose_landmarker_lite` "
            f"(sama dengan web app), maks {config.MAX_FPS} fps, sisi terpanjang {config.MAX_SIDE_PX}px.",
            "",
            f"{len(videos)} video dengan keypoint ({n_missing} video tanpa keypoint, lihat `errors.csv`). "
            "Visibility = rata-rata kiri+kanan pada frame dengan pose terdeteksi. Diurutkan dari deteksi terburuk.",
            "",
            header,
            "|---|---:|---:|---:|" + "---:|" * len(JOINT_GROUPS) + "---|",
            *rows,
            "",
            "## Temuan",
            "",
            *[f"{i}. {text}" for i, text in enumerate(findings(table, videos), 1)],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=config.REPORTS_DIR / "00-data")
    args = parser.parse_args()

    manifest = pd.read_csv(config.MANIFEST_PATH, keep_default_na=False)
    videos = load_video_stats(manifest, config.KEYPOINTS_DIR)
    table = aggregate_by_class(videos)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out_dir / "pose_quality.csv", index=False)
    (args.out_dir / "pose_quality.md").write_text(
        render_markdown(table, videos, len(manifest) - len(videos)), encoding="utf-8"
    )
    print(f"{len(videos)} videos, {len(table)} classes -> {args.out_dir}")


if __name__ == "__main__":
    main()
