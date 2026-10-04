"""Write the parity fixture shared by the Python and JS feature implementations.

    python -m repcount.features.golden                  # deterministic synthetic sequence (default)
    python -m repcount.features.golden --video-id push_up_17

The default input is synthetic so the fixture can be regenerated and checked in CI without the
4.6 GB dataset, and so it can deliberately contain the edge cases a real clip rarely holds:
a short NaN gap that gets interpolated, a long one that does not, a frame whose torso scale
collapses, permanently low ankle visibility, and a non-square frame (docs/FEATURES.md §10).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from repcount import config
from repcount.features import features as feat

FIXTURE_PATH = config.REPO_ROOT / "app" / "tests" / "fixtures" / "features_golden.json"

SYNTHETIC = {
    "n_frames": 96,
    "src_fps": 30.0,
    "width": 1280,
    "height": 720,
    "reps_per_second": 0.5,
    "short_gap": (20, 23),      # ≤ 5 resampled frames → interpolated (§7)
    "long_gap": (50, 70),       # > 5 resampled frames → stays missing, drops window 1 (§8)
    "collapsed_scale": 80,      # shoulders land on the hips → scale < MIN_SCALE (§3)
    "ankle_visibility": 0.2,
}


# ---------------------------------------------------------------- synthetic input


def _squat_pose(depth: float) -> dict[int, tuple[float, float]]:
    """MediaPipe landmark index → (x, y) for a side-on squat at `depth` in [0, 1]."""
    hip_y = 0.55 + 0.12 * depth
    shoulder_y = 0.25 + 0.16 * depth
    lean = 0.04 * depth
    return {
        0: (0.50 + lean, 0.10 + 0.18 * depth),
        11: (0.45 + lean, shoulder_y), 12: (0.55 + lean, shoulder_y),
        13: (0.43 + lean, shoulder_y + 0.14), 14: (0.57 + lean, shoulder_y + 0.14),
        15: (0.42 + lean, shoulder_y + 0.28), 16: (0.58 + lean, shoulder_y + 0.28),
        23: (0.46, hip_y), 24: (0.54, hip_y),
        25: (0.45 + 0.03 * depth, 0.76), 26: (0.55 + 0.03 * depth, 0.76),
        27: (0.45, 0.95), 28: (0.55, 0.95),
    }


def synthetic_sequence(spec: dict = SYNTHETIC, seed: int = config.SEED) -> dict:
    """Deterministic (T, 33, 4) pose sequence with the edge cases listed in the module docstring."""
    rng = np.random.default_rng(seed)
    n, fps = spec["n_frames"], spec["src_fps"]
    landmarks = np.full((n, config.N_LANDMARKS, 4), np.nan, np.float64)

    for i in range(n):
        phase = 2 * math.pi * spec["reps_per_second"] * i / fps
        pose = _squat_pose((1 - math.cos(phase)) / 2)
        for index, (x, y) in pose.items():
            jitter = rng.normal(0.0, 0.0015, 2)
            visibility = spec["ankle_visibility"] if index in (27, 28) else 0.9
            landmarks[i, index] = (x + jitter[0], y + jitter[1], 0.0, visibility)
        if i == spec["collapsed_scale"]:
            # After the jitter, so both torso lengths are exactly 0 and the scale really does collapse.
            landmarks[i, 11, :2] = landmarks[i, 23, :2]
            landmarks[i, 12, :2] = landmarks[i, 24, :2]

    for start, end in (spec["short_gap"], spec["long_gap"]):
        landmarks[start:end] = np.nan

    return {
        "landmarks": np.round(landmarks, 6),
        "timestamps_ms": np.round(np.arange(n) * 1000 / fps).astype(np.int64),
        "width": spec["width"],
        "height": spec["height"],
        "source": "synthetic",
    }


def keypoint_sequence(video_id: str, keypoints_dir: Path) -> dict:
    from repcount.features.windows import find_keypoint_files, load_sequence

    files = find_keypoint_files(keypoints_dir)
    if video_id not in files:
        raise SystemExit(f"no keypoints for {video_id} in {keypoints_dir}")
    data = load_sequence(files[video_id])
    return {
        "landmarks": np.round(np.asarray(data["landmarks"], np.float64), 6),
        "timestamps_ms": np.asarray(data["timestamps_ms"], np.int64),
        "width": int(data["width"]),
        "height": int(data["height"]),
        "source": video_id,
    }


# ---------------------------------------------------------------- fixture


def _jsonable(array: np.ndarray, decimals: int = 6) -> list:
    """NaN has no JSON literal, so it travels as null; both implementations read null back as NaN."""
    rounded = np.round(np.asarray(array, dtype=np.float64), decimals)
    return [None if not math.isfinite(v) else v for v in rounded.ravel()] if rounded.ndim == 1 else [
        _jsonable(row, decimals) for row in rounded
    ]


def build_fixture(sequence: dict) -> dict:
    """Run every stage of docs/FEATURES.md on `sequence` and record the expected output of each."""
    indices = feat.resample_indices(sequence["timestamps_ms"])
    features = feat.sequence_features(
        sequence["landmarks"], sequence["timestamps_ms"], sequence["width"], sequence["height"]
    )
    cut = feat.make_windows(features)
    return {
        "spec": "docs/FEATURES.md",
        "generator": "python -m repcount.features.golden",
        "source": sequence["source"],
        "tolerance": 1e-4,
        "constants": {
            "targetFps": feat.TARGET_FPS,
            "windowFrames": feat.WINDOW_FRAMES,
            "windowStride": feat.WINDOW_STRIDE,
            "maxInterpolationGap": feat.MAX_INTERPOLATION_GAP,
            "maxMissingRatio": feat.MAX_MISSING_RATIO,
            "nFeatures": feat.N_FEATURES,
        },
        "input": {
            "landmarks": _jsonable(sequence["landmarks"]),
            "timestampsMs": [int(t) for t in sequence["timestamps_ms"]],
            "width": int(sequence["width"]),
            "height": int(sequence["height"]),
        },
        "expected": {
            "resampleIndices": [int(i) for i in indices],
            "sequenceFeatures": _jsonable(features),
            "missing": [bool(m) for m in feat.missing_frames(features)],
            "windowStarts": [int(s) for s in cut["starts"]],
            "missingRatio": _jsonable(cut["missing_ratio"]),
            "usable": [bool(u) for u in feat.usable(cut["missing_ratio"])],
            "windows": _jsonable(cut["windows"]),
            "flippedFirstWindow": _jsonable(feat.flip_features(cut["windows"][0])) if len(cut["windows"]) else [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-id", default=None, help="use a real extracted video instead of synthetic input")
    parser.add_argument("--keypoints-dir", type=Path, default=config.KEYPOINTS_DIR)
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()

    sequence = keypoint_sequence(args.video_id, args.keypoints_dir) if args.video_id else synthetic_sequence()
    fixture = build_fixture(sequence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    expected = fixture["expected"]
    print(f"{args.out} ({args.out.stat().st_size / 1024:.0f} KB) from {fixture['source']}: "
          f"{len(expected['sequenceFeatures'])} frames, {len(expected['windows'])} windows, "
          f"usable {sum(expected['usable'])}/{len(expected['usable'])}")


if __name__ == "__main__":
    main()
