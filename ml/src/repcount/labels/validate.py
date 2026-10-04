"""Check the human repetition labels and summarise them per class.

    python -m repcount.labels.validate                 # labels/rep_labels.csv against the manifest
    python -m repcount.labels.validate --agreement     # + compare with labels/rep_labels_recheck.csv

Exits with status 1 when any label breaks a rule in labels/README.md, so it can gate a commit.
`--agreement` measures how consistently the same person labels the same video twice — the ceiling a
rep counter can reasonably be held to in stage 3.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from repcount import config
from repcount.labels.select import LABELS_DIR, TO_LABEL_PATH, task_for

LABELS_PATH = LABELS_DIR / "rep_labels.csv"
RECHECK_PATH = LABELS_DIR / "rep_labels_recheck.csv"

REP_LABEL_COLUMNS = [
    "video_id", "label", "rep_count", "rep_timestamps_ms", "hold_start_ms", "hold_end_ms",
    "is_ambiguous", "notes", "labeler", "labeled_at",
]
# The browser's video duration and the manifest's n_frames / fps can disagree by a frame or two.
DURATION_TOLERANCE_MS = 250
# Two marks of the same rep in different sessions agree when they are at most this far apart.
AGREEMENT_TOLERANCE_MS = 300


# ---------------------------------------------------------------- parsing


def parse_timestamps(text: str) -> list[int]:
    """'1200;2400' → [1200, 2400]; '' → []. Raises ValueError on anything that is not an integer."""
    return [int(part) for part in str(text).split(";") if part.strip() != ""]


def _int_or_none(text: str) -> int | None:
    return None if str(text).strip() == "" else int(text)


def _is_timestamp(text: str) -> bool:
    try:
        datetime.fromisoformat(str(text))
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------- per-row rules


def _common_problems(row: dict, manifest_row: dict | None) -> list[str]:
    problems = []
    if manifest_row is None:
        return [f"video_id {row['video_id']!r} tidak ada di manifest"]
    if row["label"] != manifest_row["label"]:
        problems.append(f"label {row['label']!r} ≠ manifest {manifest_row['label']!r}")
    if row["is_ambiguous"] not in ("true", "false"):
        problems.append(f"is_ambiguous harus 'true' atau 'false', bukan {row['is_ambiguous']!r}")
    elif row["is_ambiguous"] == "true" and not row["notes"].strip():
        problems.append("ditandai ambigu tanpa catatan")
    if not row["labeler"].strip():
        problems.append("labeler kosong")
    if not _is_timestamp(row["labeled_at"]):
        problems.append(f"labeled_at bukan waktu ISO 8601: {row['labeled_at']!r}")
    return problems


def _rep_problems(row: dict, duration_ms: float) -> list[str]:
    try:
        count, stamps = _int_or_none(row["rep_count"]), parse_timestamps(row["rep_timestamps_ms"])
    except ValueError:
        return ["rep_count atau rep_timestamps_ms bukan bilangan bulat"]
    problems = []
    if count is None or count != len(stamps):
        problems.append(f"rep_count {row['rep_count']!r} ≠ jumlah timestamp ({len(stamps)})")
    if any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
        problems.append("timestamp tidak naik atau ada duplikat")
    if stamps and (stamps[0] < 0 or stamps[-1] > duration_ms + DURATION_TOLERANCE_MS):
        problems.append(f"timestamp di luar durasi video ({duration_ms:.0f} ms)")
    if row["hold_start_ms"].strip() or row["hold_end_ms"].strip():
        problems.append("hold_start_ms/hold_end_ms harus kosong untuk latihan repetisi")
    if count == 0 and row["is_ambiguous"] != "true":
        problems.append("0 rep tanpa ditandai ambigu — setiap video terpilih seharusnya berisi latihan")
    return problems


def _hold_problems(row: dict, duration_ms: float) -> list[str]:
    problems = []
    if row["rep_count"].strip() or row["rep_timestamps_ms"].strip():
        problems.append("plank dilabel sebagai durasi tahan: rep_count & rep_timestamps_ms harus kosong")
    try:
        start, end = _int_or_none(row["hold_start_ms"]), _int_or_none(row["hold_end_ms"])
    except ValueError:
        return [*problems, "hold_start_ms atau hold_end_ms bukan bilangan bulat"]
    if start is None or end is None:
        if row["is_ambiguous"] != "true":
            problems.append("hold_start_ms dan hold_end_ms wajib diisi")
    elif not 0 <= start < end <= duration_ms + DURATION_TOLERANCE_MS:
        problems.append(f"butuh 0 ≤ mulai < selesai ≤ durasi ({duration_ms:.0f} ms), dapat {start}–{end}")
    return problems


def row_problems(row: dict, manifest_row: dict | None) -> list[str]:
    """Every rule in labels/README.md that one label row breaks."""
    problems = _common_problems(row, manifest_row)
    if manifest_row is None:
        return problems
    duration_ms = float(manifest_row["duration_s"]) * 1000
    task_rules = _hold_problems if task_for(manifest_row["label"]) == "hold" else _rep_problems
    return problems + task_rules(row, duration_ms)


def validate_labels(labels: pd.DataFrame, manifest: pd.DataFrame) -> list[tuple[str, str]]:
    """(video_id, problem) for every broken rule, including missing columns and duplicate rows."""
    missing = [c for c in REP_LABEL_COLUMNS if c not in labels.columns]
    if missing:
        return [("-", f"kolom hilang: {', '.join(missing)}")]
    by_id = {row["video_id"]: row for row in manifest.to_dict("records")}
    problems = [(vid, "video_id muncul lebih dari sekali")
                for vid in sorted(labels["video_id"][labels["video_id"].duplicated()].unique())]
    for row in labels.to_dict("records"):
        problems += [(row["video_id"], p) for p in row_problems(row, by_id.get(row["video_id"]))]
    return problems


# ---------------------------------------------------------------- summaries


def class_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Per class: labelled videos, ambiguous ones, and the rep (or hold-seconds) distribution."""
    rows = []
    for label, group in labels.groupby("label", sort=True):
        if task_for(label) == "hold":
            values = [(int(e) - int(s)) / 1000 for s, e in zip(group["hold_start_ms"], group["hold_end_ms"],
                                                               strict=True) if str(s).strip() and str(e).strip()]
            unit = "detik tahan"
        else:
            values = [len(parse_timestamps(t)) for t in group["rep_timestamps_ms"]]
            unit = "rep"
        rows.append({
            "label": label, "n_videos": len(group), "n_ambiguous": int((group["is_ambiguous"] == "true").sum()),
            "unit": unit, "mean": sum(values) / len(values) if values else float("nan"),
            "min": min(values) if values else float("nan"), "max": max(values) if values else float("nan"),
        })
    return pd.DataFrame(rows)


def unlabelled(labels: pd.DataFrame, to_label: pd.DataFrame) -> list[str]:
    return sorted(set(to_label["video_id"]) - set(labels["video_id"]))


# ---------------------------------------------------------------- agreement between sessions


def matched_marks(first: list[int], second: list[int], tolerance: int = AGREEMENT_TOLERANCE_MS) -> int:
    """Marks pairable one-to-one within `tolerance` ms (both lists ascending, greedy two-pointer)."""
    i = j = matched = 0
    while i < len(first) and j < len(second):
        if abs(first[i] - second[j]) <= tolerance:
            matched, i, j = matched + 1, i + 1, j + 1
        elif first[i] < second[j]:
            i += 1
        else:
            j += 1
    return matched


def agreement(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    """Count and timing agreement on the videos labelled in both sessions (rep videos only)."""
    a = first.set_index("video_id")
    b = second.set_index("video_id")
    common = [v for v in sorted(set(a.index) & set(b.index)) if task_for(a.at[v, "label"]) == "reps"]
    per_video = []
    for video in common:
        stamps_a, stamps_b = parse_timestamps(a.at[video, "rep_timestamps_ms"]), parse_timestamps(
            b.at[video, "rep_timestamps_ms"])
        per_video.append({
            "video_id": video, "first": len(stamps_a), "second": len(stamps_b),
            "difference": len(stamps_b) - len(stamps_a),
            "timing_match": matched_marks(stamps_a, stamps_b) / max(len(stamps_a), len(stamps_b), 1),
        })
    if not per_video:
        return {"n_videos": 0, "per_video": []}
    differences = [abs(v["difference"]) for v in per_video]
    return {
        "n_videos": len(per_video),
        "exact": sum(d == 0 for d in differences) / len(differences),
        "off_by_one": sum(d <= 1 for d in differences) / len(differences),
        "mae": sum(differences) / len(differences),
        "timing_match": sum(v["timing_match"] for v in per_video) / len(per_video),
        "per_video": per_video,
    }


# ---------------------------------------------------------------- I/O


def read_labels(path: Path) -> pd.DataFrame:
    """Every column as text: empty cells stay '' instead of becoming NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def print_summary(labels: pd.DataFrame, to_label: pd.DataFrame | None) -> None:
    summary = class_summary(labels)
    print(f"{len(labels)} video berlabel, {int(summary['n_ambiguous'].sum())} ambigu\n")
    print(f"{'kelas':<22} {'video':>5} {'ambigu':>6}  {'rata-rata':>9} {'min':>5} {'maks':>5}  satuan")
    for row in summary.to_dict("records"):
        print(f"{row['label']:<22} {row['n_videos']:>5} {row['n_ambiguous']:>6}  "
              f"{row['mean']:>9.1f} {row['min']:>5g} {row['max']:>5g}  {row['unit']}")
    if to_label is not None:
        todo = unlabelled(labels, to_label)
        print(f"\nbelum dilabel: {len(todo)} dari {len(to_label)} video di to_label.csv"
              + (f" ({', '.join(todo[:8])}{', …' if len(todo) > 8 else ''})" if todo else ""))


def print_agreement(result: dict) -> None:
    if not result["n_videos"]:
        print("\nkonsistensi: tidak ada video repetisi yang dilabel di kedua sesi")
        return
    print(f"\nkonsistensi antar sesi ({result['n_videos']} video): jumlah sama persis {result['exact']:.0%}, "
          f"selisih ≤ 1 {result['off_by_one']:.0%}, MAE {result['mae']:.2f} rep, "
          f"waktu tanda cocok (±{AGREEMENT_TOLERANCE_MS} ms) {result['timing_match']:.0%}")
    for video in result["per_video"]:
        if video["difference"]:
            print(f"  {video['video_id']}: {video['first']} → {video['second']} rep")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--to-label", type=Path, default=TO_LABEL_PATH)
    parser.add_argument("--agreement", action="store_true", help="compare with --recheck")
    parser.add_argument("--recheck", type=Path, default=RECHECK_PATH)
    args = parser.parse_args()

    for path in (args.labels, args.manifest):
        if not path.exists():
            raise SystemExit(f"{path} not found")
    labels = read_labels(args.labels)
    manifest = pd.read_csv(args.manifest, keep_default_na=False)

    sessions = [("rep_labels", labels)]
    if args.agreement:
        if not args.recheck.exists():
            raise SystemExit(f"{args.recheck} not found — label the recheck subset first")
        sessions.append(("recheck", read_labels(args.recheck)))

    problems = [(name, vid, p) for name, frame in sessions for vid, p in validate_labels(frame, manifest)]
    for name, video_id, problem in problems:
        print(f"[{name}] {video_id}: {problem}")
    if problems:
        print(f"\n{len(problems)} masalah ditemukan — perbaiki di alat labeling, lalu export ulang.")
        return 1

    print_summary(labels, read_labels(args.to_label) if args.to_label.exists() else None)
    if args.agreement:
        print_agreement(agreement(labels, sessions[1][1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
