"""Global paths and constants. All data lives under one `data/` folder at the repo root."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("FITNESS_DATA_DIR", REPO_ROOT / "data")).resolve()

VIDEOS_DIR = DATA_DIR / "workout-videos"
KEYPOINTS_DIR = DATA_DIR / "keypoints"
POSE_MODELS_DIR = DATA_DIR / "pose_models"
RUNS_DIR = DATA_DIR / "runs"

MANIFEST_PATH = KEYPOINTS_DIR / "manifest.csv"
ERRORS_PATH = KEYPOINTS_DIR / "errors.csv"

SPLITS_DIR = REPO_ROOT / "ml" / "splits"
REPORTS_DIR = REPO_ROOT / "reports"

# Must stay identical to POSE_MODEL_URL in app/src/config.js (train/serve consistency).
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = POSE_MODELS_DIR / "pose_landmarker_lite.task"

# The web app does not set confidences, so it runs on the MediaPipe defaults (0.5).
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

VIDEO_EXTENSIONS = (".mp4", ".mov")
MAX_FPS = 30
MAX_SIDE_PX = 640
N_LANDMARKS = 33

SEED = 42
