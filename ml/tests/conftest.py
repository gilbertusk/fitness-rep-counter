from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tiny_video():
    """Factory writing a small synthetic MJPG video; returns its path."""

    def make(path: Path, n_frames: int = 10, fps: float = 10.0, size: tuple[int, int] = (64, 48)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        rng = np.random.default_rng(0)
        for _ in range(n_frames):
            writer.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        writer.release()
        return path

    return make
