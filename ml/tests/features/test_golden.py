"""Python side of the feature parity test (docs/FEATURES.md §10).

`app/tests/unit/features/features.test.js` asserts the same fixture from the JS side; together they
pin both implementations to one set of numbers.
"""

import json

import numpy as np
import pytest

from repcount.features import features as feat
from repcount.features.golden import FIXTURE_PATH, build_fixture, synthetic_sequence


@pytest.fixture(scope="module")
def fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.fail(f"{FIXTURE_PATH} is missing — run `python -m repcount.features.golden`")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def as_array(nested) -> np.ndarray:
    """JSON null (NaN has no literal) → np.nan."""
    return np.array(nested, dtype=object).astype(np.float64) if nested else np.array([])


def test_fixture_constants_match_the_implementation(fixture):
    constants = fixture["constants"]
    assert constants["targetFps"] == feat.TARGET_FPS
    assert constants["windowFrames"] == feat.WINDOW_FRAMES
    assert constants["windowStride"] == feat.WINDOW_STRIDE
    assert constants["maxInterpolationGap"] == feat.MAX_INTERPOLATION_GAP
    assert constants["maxMissingRatio"] == feat.MAX_MISSING_RATIO
    assert constants["nFeatures"] == feat.N_FEATURES


def test_pipeline_reproduces_the_golden_features(fixture):
    source = fixture["input"]
    features = feat.sequence_features(
        as_array(source["landmarks"]), np.array(source["timestampsMs"]), source["width"], source["height"]
    )
    expected = as_array(fixture["expected"]["sequenceFeatures"])
    assert features.shape == expected.shape
    assert features == pytest.approx(expected, abs=fixture["tolerance"], nan_ok=True)


def test_pipeline_reproduces_the_golden_windows(fixture):
    source = fixture["input"]
    features = feat.sequence_features(
        as_array(source["landmarks"]), np.array(source["timestampsMs"]), source["width"], source["height"]
    )
    cut = feat.make_windows(features)
    expected = fixture["expected"]
    assert cut["starts"].tolist() == expected["windowStarts"]
    assert feat.usable(cut["missing_ratio"]).tolist() == expected["usable"]
    assert cut["missing_ratio"] == pytest.approx(as_array(expected["missingRatio"]), abs=fixture["tolerance"])
    assert cut["windows"] == pytest.approx(as_array(expected["windows"]), abs=fixture["tolerance"])
    assert feat.flip_features(cut["windows"][0]) == pytest.approx(
        as_array(expected["flippedFirstWindow"]), abs=fixture["tolerance"]
    )


def test_fixture_exercises_the_edge_cases_it_claims_to(fixture):
    missing = fixture["expected"]["missing"]
    assert any(missing), "no missing frame — the long NaN gap is not in the fixture"
    assert not all(missing)
    assert fixture["expected"]["usable"] == [True, False], "fixture must cover a kept and a dropped window"
    visibility_columns = fixture["expected"]["sequenceFeatures"][0][2:39:3]
    assert min(visibility_columns) < 0.5, "fixture must cover a landmark with low visibility"


def test_fixture_on_disk_is_up_to_date(fixture):
    """Guards against editing features.py without regenerating the fixture."""
    rebuilt = build_fixture(synthetic_sequence())
    assert rebuilt["expected"]["sequenceFeatures"] == fixture["expected"]["sequenceFeatures"]
    assert rebuilt["expected"]["windows"] == fixture["expected"]["windows"]


def test_synthetic_sequence_is_deterministic():
    first, second = synthetic_sequence()["landmarks"], synthetic_sequence()["landmarks"]
    assert first == pytest.approx(second, nan_ok=True)


def test_keypoint_sequence_reads_an_extracted_video(tmp_path):
    from repcount.features.golden import keypoint_sequence

    sequence = synthetic_sequence()
    path = tmp_path / "squat" / "s_1.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        landmarks=sequence["landmarks"].astype(np.float32),
        timestamps_ms=sequence["timestamps_ms"],
        width=np.int32(1920),
        height=np.int32(1080),
        label=np.str_("squat"),
        video_id=np.str_("s_1"),
    )
    loaded = keypoint_sequence("s_1", tmp_path)
    assert loaded["source"] == "s_1"
    assert (loaded["width"], loaded["height"]) == (1920, 1080)
    assert loaded["landmarks"].shape == sequence["landmarks"].shape


def test_keypoint_sequence_fails_loudly_for_an_unknown_video(tmp_path):
    from repcount.features.golden import keypoint_sequence

    with pytest.raises(SystemExit, match="no keypoints for nope_1"):
        keypoint_sequence("nope_1", tmp_path)
