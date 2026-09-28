"""Extract MediaPipe pose keypoints for every video in the manifest.

Uses the same model as the web app (pose_landmarker_lite, VIDEO mode, one pose, default 0.5
confidences). Frames are sampled to at most MAX_FPS and resized so the longest side is at most
MAX_SIDE_PX before inference. Output: KEYPOINTS_DIR/<label>/<video_id>.npz. Resumable.

    python -m repcount.data.extract [--limit N] [--workers N] [--overwrite]
"""

import argparse
import math
import os
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from repcount import config

FPS_TOLERANCE = 1.01  # 30.05 fps sources are kept whole instead of dropping one frame in 600


# ---------------------------------------------------------------- pure helpers


def keep_frame(index: int, src_fps: float, max_fps: float) -> bool:
    """Streaming frame sampler: keep frame `index` of a `src_fps` video so output is <= max_fps."""
    if src_fps <= max_fps * FPS_TOLERANCE or index == 0:
        return True
    ratio = max_fps / src_fps
    return math.floor(index * ratio) != math.floor((index - 1) * ratio)


def sample_frame_indices(n_frames: int, src_fps: float, max_fps: float) -> list[int]:
    return [i for i in range(n_frames) if keep_frame(i, src_fps, max_fps)]


def resize_dims(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Size with the longest side <= max_side, aspect ratio kept, never upscaled."""
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def timestamp_ms(index: int, src_fps: float) -> int:
    return round(index * 1000 / src_fps)


def pose_to_arrays(result) -> tuple[np.ndarray, np.ndarray]:
    """First pose of a PoseLandmarkerResult → (33, 4) landmarks and (33, 3) world landmarks, NaN if none."""
    if not result.pose_landmarks:
        return (
            np.full((config.N_LANDMARKS, 4), np.nan, np.float32),
            np.full((config.N_LANDMARKS, 3), np.nan, np.float32),
        )
    landmarks = np.array([[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]], np.float32)
    world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]], np.float32)
    return landmarks, world


def output_path(keypoints_dir: Path, label: str, video_id: str) -> Path:
    return keypoints_dir / label / f"{video_id}.npz"


# ---------------------------------------------------------------- I/O


def create_landmarker(model_path: Path):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return vision.PoseLandmarker.create_from_options(options)


def iter_frames(cap) -> Iterator[tuple[int, np.ndarray]]:
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield index, frame
        index += 1


def extract_video(video_path: Path, model_path: Path) -> dict:
    """Run pose detection on one video; returns the arrays to store (without label/id)."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        raise RuntimeError(f"invalid fps {src_fps}")

    landmarks, world, stamps, size, n_decoded = [], [], [], None, 0
    try:
        with create_landmarker(model_path) as landmarker:
            for index, frame in iter_frames(cap):
                n_decoded = index + 1
                if not keep_frame(index, src_fps, config.MAX_FPS):
                    continue
                size = size or (frame.shape[1], frame.shape[0])
                rgb = cv2.cvtColor(
                    cv2.resize(frame, resize_dims(*size, config.MAX_SIDE_PX), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                stamp = timestamp_ms(index, src_fps)
                lm, wl = pose_to_arrays(landmarker.detect_for_video(image, stamp))
                landmarks.append(lm)
                world.append(wl)
                stamps.append(stamp)
    finally:
        cap.release()

    if not stamps:
        raise RuntimeError("no frame decoded")
    return {
        "landmarks": np.stack(landmarks),
        "world_landmarks": np.stack(world),
        "timestamps_ms": np.array(stamps, np.int64),
        "fps_effective": np.float32(len(stamps) * src_fps / n_decoded),
        "width": np.int32(size[0]),
        "height": np.int32(size[1]),
    }


def save_npz(path: Path, arrays: dict) -> None:
    """Atomic write so an interrupted run never leaves a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def process_one(row: dict, videos_dir: str, keypoints_dir: str, model_path: str) -> dict:
    """Worker entry point. Never raises: returns a status record."""
    start = time.perf_counter()
    out = output_path(Path(keypoints_dir), row["label"], row["video_id"])
    try:
        arrays = extract_video(Path(videos_dir) / row["path"], Path(model_path))
        save_npz(out, {**arrays, "label": np.str_(row["label"]), "video_id": np.str_(row["video_id"])})
        return {
            **row,
            "ok": True,
            "n_frames": len(arrays["timestamps_ms"]),
            "seconds": time.perf_counter() - start,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - one bad video must not stop the batch
        return {
            **row,
            "ok": False,
            "n_frames": 0,
            "seconds": time.perf_counter() - start,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------- CLI


def select_pending(manifest: pd.DataFrame, keypoints_dir: Path, overwrite: bool, limit: int | None) -> list[dict]:
    rows = manifest[["video_id", "label", "path"]].to_dict("records")
    if not overwrite:
        rows = [r for r in rows if not output_path(keypoints_dir, r["label"], r["video_id"]).exists()]
    return rows[:limit] if limit else rows


def write_errors(results: list[dict], errors_path: Path) -> int:
    failed = [{k: r[k] for k in ("video_id", "label", "path", "error")} for r in results if not r["ok"]]
    previous = pd.read_csv(errors_path, keep_default_na=False) if errors_path.exists() else pd.DataFrame()
    retried = {r["video_id"] for r in results}
    kept = previous[~previous["video_id"].isin(retried)] if len(previous) else previous
    pd.concat([kept, pd.DataFrame(failed, columns=["video_id", "label", "path", "error"])]).to_csv(
        errors_path, index=False
    )
    return len(failed)


def run(rows: list[dict], workers: int, videos_dir: Path, keypoints_dir: Path, model_path: Path) -> list[dict]:
    results, start = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, r, str(videos_dir), str(keypoints_dir), str(model_path)) for r in rows]
        for done, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = f"{r['n_frames']} frames" if r["ok"] else f"ERROR {r['error']}"
            print(
                f"[{done}/{len(rows)}] {time.perf_counter() - start:7.1f}s  {r['video_id']}: "
                f"{status} ({r['seconds']:.1f}s)",
                flush=True,
            )
    return results


def main() -> None:
    from repcount.data.download_model import download_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="process at most N pending videos")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--overwrite", action="store_true", help="re-extract videos that already have a .npz")
    args = parser.parse_args()

    model_path = download_model()
    manifest = pd.read_csv(config.MANIFEST_PATH, keep_default_na=False)
    rows = select_pending(manifest, config.KEYPOINTS_DIR, args.overwrite, args.limit)
    print(f"{len(rows)} videos to process with {args.workers} workers", flush=True)

    start = time.perf_counter()
    results = run(rows, args.workers, config.VIDEOS_DIR, config.KEYPOINTS_DIR, model_path)
    n_failed = write_errors(results, config.ERRORS_PATH)
    total = sum(1 for _ in config.KEYPOINTS_DIR.glob("*/*.npz"))
    print(
        f"Done in {time.perf_counter() - start:.1f}s: {len(results) - n_failed} ok, {n_failed} failed "
        f"(see {config.ERRORS_PATH}). {total} .npz files in total."
    )


if __name__ == "__main__":
    main()
