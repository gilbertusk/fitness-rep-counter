"""Download the MediaPipe pose model used by the web app.

    python -m repcount.data.download_model [--overwrite]
"""

import argparse
import urllib.request
from pathlib import Path

from repcount.config import POSE_MODEL_PATH, POSE_MODEL_URL


def download_model(dest: Path = POSE_MODEL_PATH, url: str = POSE_MODEL_URL, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL from config
    tmp.replace(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = download_model(overwrite=args.overwrite)
    print(f"Model: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
