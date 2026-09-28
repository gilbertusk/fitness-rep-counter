"""Near-duplicate detection with a 64-bit difference hash (dHash) and union-find grouping.

Each video is fingerprinted by the dHash of the frames at 10%, 50% and 90% of its duration
(dark letterbox bars cropped first).
Two videos are near-duplicates when the mean Hamming distance over those three frames is
<= --threshold (default 10 of 64 bits). Near-duplicates share a `group_id`, written to the manifest.

Threshold choice (docs/PLAN.md §8): started at 6; spot checks of pairs at 6-10 all showed the same
person and scene (clips cut from one source video), and no cross-class pair appears up to 10.
The first cross-class pair appears at 12, so 10 is the largest threshold without obvious false positives.

    python -m repcount.data.dedup [--threshold 6]
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from repcount.config import MANIFEST_PATH, VIDEOS_DIR

DEFAULT_THRESHOLD = 10.0
SAMPLE_POSITIONS = (0.1, 0.5, 0.9)
HASH_SIZE = 8  # 8x8 comparisons → 64 bits
BORDER_LEVEL = 16  # rows/cols whose brightest pixel is below this are letterbox bars


def crop_borders(gray: np.ndarray, level: int = BORDER_LEVEL) -> np.ndarray:
    """Remove dark letterbox/pillarbox bars; vertical phone clips padded to 16:9 would otherwise
    hash alike because the bars dominate the image. Returns the input unchanged if it is all dark."""
    rows = np.flatnonzero(gray.max(axis=1) >= level)
    cols = np.flatnonzero(gray.max(axis=0) >= level)
    if rows.size == 0 or cols.size == 0:
        return gray
    return gray[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash: crop dark bars, shrink to 9x8, bit = pixel brighter than its right neighbour."""
    import cv2

    small = cv2.resize(crop_borders(gray), (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def mean_distance_matrix(hashes: np.ndarray) -> np.ndarray:
    """(N, K) uint64 hashes → (N, N) mean Hamming distance over the K sampled frames."""
    xor = hashes[:, None, :] ^ hashes[None, :, :]
    return np.bitwise_count(xor).mean(axis=2)


def find_duplicate_pairs(hashes: np.ndarray, valid: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Pairs (i, j, distance) with i < j, both valid, and mean distance <= threshold."""
    dist = mean_distance_matrix(hashes)
    close = (dist <= threshold) & valid[:, None] & valid[None, :]
    rows, cols = np.nonzero(np.triu(close, k=1))
    return [(int(i), int(j), float(dist[i, j])) for i, j in zip(rows, cols, strict=True)]


def union_find_groups(n: int, pairs: Sequence[tuple[int, int]]) -> list[int]:
    """Connected components of n items; returns a component index per item, numbered by first appearance."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    numbering: dict[int, int] = {}
    return [numbering.setdefault(find(i), len(numbering)) for i in range(n)]


def format_group_ids(components: Sequence[int]) -> list[str]:
    return [f"g{c:04d}" for c in components]


def video_hashes(path: Path, positions: Sequence[float] = SAMPLE_POSITIONS) -> list[int] | None:
    """dHash of the frames at the given relative positions, or None if the video cannot be read."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or n_frames <= 0:
            return None
        hashes = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos * (n_frames - 1)))
            ok, frame = cap.read()
            if not ok:
                return None
            hashes.append(dhash(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        return hashes
    finally:
        cap.release()


def threshold_sensitivity(hashes: np.ndarray, valid: np.ndarray, thresholds: Sequence[float]) -> dict:
    """Number of videos that end up in a multi-video group, per candidate threshold."""
    result = {}
    for t in thresholds:
        pairs = [(i, j) for i, j, _ in find_duplicate_pairs(hashes, valid, t)]
        sizes = np.bincount(union_find_groups(len(hashes), pairs))
        result[t] = int(sizes[sizes > 1].sum())
    return result


def assign_groups(manifest: pd.DataFrame, videos_dir: Path, threshold: float) -> tuple[pd.DataFrame, list, dict]:
    hashes = [video_hashes(videos_dir / p) for p in manifest["path"]]
    valid = np.array([h is not None for h in hashes])
    matrix = np.array([h or [0] * len(SAMPLE_POSITIONS) for h in hashes], dtype=np.uint64)
    pairs = find_duplicate_pairs(matrix, valid, threshold)
    groups = format_group_ids(union_find_groups(len(manifest), [(i, j) for i, j, _ in pairs]))
    sensitivity = threshold_sensitivity(matrix, valid, (2, 4, 6, 8, 10, 12))
    return manifest.assign(group_id=groups), pairs, sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, keep_default_na=False).drop(columns="group_id", errors="ignore")
    updated, pairs, sensitivity = assign_groups(manifest, args.videos_dir, args.threshold)
    updated.to_csv(args.manifest, index=False)

    sizes = updated["group_id"].value_counts()
    print(
        f"threshold={args.threshold}: {len(pairs)} duplicate pairs, {int((sizes > 1).sum())} groups with >1 video, "
        f"{int(sizes[sizes > 1].sum())} videos in them, {updated['group_id'].nunique()} groups total"
    )
    print("videos in multi-video groups per threshold:", sensitivity)
    for i, j, d in pairs:
        print(f"  {d:5.2f}  {updated.at[i, 'path']}  <->  {updated.at[j, 'path']}")


if __name__ == "__main__":
    main()
