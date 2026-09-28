"""Scan the raw dataset and write KEYPOINTS_DIR/manifest.csv.

    python -m repcount.data.manifest
"""

import argparse
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from repcount.config import MANIFEST_PATH, VIDEO_EXTENSIONS, VIDEOS_DIR

MANIFEST_COLUMNS = [
    "video_id", "label", "path", "ext", "fps", "n_frames", "duration_s",
    "width", "height", "size_bytes", "error",
]


def normalize_label(name: str) -> str:
    """Dataset folder name → snake_case label, e.g. 'pull Up' → 'pull_up', 'push-up' → 'push_up'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def make_video_id(path: Path) -> str:
    """File stem → snake_case id, e.g. 'push-up_17.mp4' → 'push_up_17'."""
    return normalize_label(Path(path).stem)


def list_videos(videos_dir: Path) -> list[dict]:
    """Every video under `videos_dir/<class>/`, sorted, with label and id (no file is opened)."""
    records = [
        {
            "video_id": make_video_id(path),
            "label": normalize_label(path.parent.name),
            "path": path.relative_to(videos_dir).as_posix(),
            "ext": path.suffix.lower().lstrip("."),
        }
        for path in sorted(videos_dir.glob("*/*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    ensure_unique_ids(records)
    return records


def ensure_unique_ids(records: Iterable[dict]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        other = seen.setdefault(record["video_id"], record["path"])
        if other != record["path"]:
            raise ValueError(f"Duplicate video_id {record['video_id']!r}: {other} and {record['path']}")


def probe_video(path: Path) -> dict:
    """Read container metadata with OpenCV. Never raises: failures go to the `error` field."""
    import cv2

    info = {"fps": None, "n_frames": None, "duration_s": None, "width": None, "height": None,
            "size_bytes": path.stat().st_size if path.exists() else None, "error": ""}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {**info, "error": "cannot open video"}
        ok, _ = cap.read()
        if not ok:
            return {**info, "error": "cannot decode first frame"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            **info,
            "fps": round(fps, 3),
            "n_frames": n_frames,
            "duration_s": round(n_frames / fps, 3) if fps > 0 else None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def build_manifest(videos_dir: Path = VIDEOS_DIR) -> pd.DataFrame:
    rows = [{**record, **probe_video(videos_dir / record["path"])} for record in list_videos(videos_dir)]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    manifest = build_manifest(args.videos_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    n_errors = int((manifest["error"] != "").sum())
    print(f"{len(manifest)} videos, {manifest['label'].nunique()} classes, {n_errors} errors -> {args.out}")


if __name__ == "__main__":
    main()
