import numpy as np

from repcount.data.dedup import (
    crop_borders,
    dhash,
    find_duplicate_pairs,
    format_group_ids,
    hamming,
    mean_distance_matrix,
    union_find_groups,
    video_hashes,
)


def gradient_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (90, 160), dtype=np.uint8)


def test_identical_images_have_distance_zero():
    img = gradient_image()
    assert hamming(dhash(img), dhash(img.copy())) == 0


def test_slightly_noisy_copy_stays_close():
    img = np.tile(np.linspace(0, 255, 160, dtype=np.uint8), (90, 1))
    img[::7] = 255 - img[::7]
    noisy = np.clip(img.astype(int) + np.random.default_rng(1).integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    assert hamming(dhash(img), dhash(noisy)) <= 6


def test_different_images_are_far_apart():
    assert hamming(dhash(gradient_image(0)), dhash(gradient_image(1))) > 16


def test_dhash_fits_in_64_bits():
    assert 0 <= dhash(gradient_image()) < 2**64


def test_hamming_counts_differing_bits():
    assert hamming(0b1011, 0b0001) == 2
    assert hamming(2**64 - 1, 0) == 64


def test_crop_borders_removes_pillarbox_bars():
    img = np.zeros((90, 160), np.uint8)
    img[:, 60:100] = 200
    assert crop_borders(img).shape == (90, 40)


def test_crop_borders_keeps_all_dark_image():
    img = np.zeros((10, 10), np.uint8)
    assert crop_borders(img).shape == (10, 10)


def test_letterboxed_different_content_is_not_a_duplicate():
    a, b = np.zeros((90, 160), np.uint8), np.zeros((90, 160), np.uint8)
    a[:, 60:100] = gradient_image(0)[:, :40]
    b[:, 60:100] = gradient_image(1)[:, :40]
    assert hamming(dhash(a), dhash(b)) > 16


def test_mean_distance_matrix_averages_over_frames():
    hashes = np.array([[0, 0, 0], [0b1, 0b11, 0b111]], dtype=np.uint64)
    dist = mean_distance_matrix(hashes)
    assert dist[0, 1] == dist[1, 0] == 2.0
    assert dist[0, 0] == 0.0


def test_find_duplicate_pairs_respects_threshold_and_validity():
    hashes = np.array([[0, 0], [0b1, 0b1], [2**40 - 1, 2**40 - 1], [0, 0]], dtype=np.uint64)
    valid = np.array([True, True, True, False])
    pairs = find_duplicate_pairs(hashes, valid, threshold=1)
    assert pairs == [(0, 1, 1.0)]


def test_union_find_joins_transitive_pairs():
    assert union_find_groups(6, [(0, 2), (2, 4), (3, 5)]) == [0, 1, 0, 2, 0, 2]


def test_union_find_without_pairs_gives_singletons():
    assert union_find_groups(3, []) == [0, 1, 2]


def test_group_ids_are_zero_padded():
    assert format_group_ids([0, 12]) == ["g0000", "g0012"]


def test_video_hashes_reads_three_frames(tmp_path, tiny_video):
    hashes = video_hashes(tiny_video(tmp_path / "v.avi", n_frames=10))
    assert hashes is not None and len(hashes) == 3


def test_video_hashes_is_none_for_unreadable_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"x")
    assert video_hashes(bad) is None


def test_threshold_sensitivity_grows_with_threshold():
    from repcount.data.dedup import threshold_sensitivity

    hashes = np.array([[0], [0b1], [0b111], [2**40 - 1]], dtype=np.uint64)
    result = threshold_sensitivity(hashes, np.ones(4, bool), (0, 1, 3))
    assert result == {0: 0, 1: 2, 3: 3}


def test_assign_groups_puts_copied_video_in_same_group(tmp_path, tiny_video):
    import shutil

    import pandas as pd

    from repcount.data.dedup import assign_groups

    tiny_video(tmp_path / "a" / "a_1.avi", n_frames=10)
    shutil.copy(tmp_path / "a" / "a_1.avi", tmp_path / "a" / "a_2.avi")
    (tmp_path / "a" / "a_3.avi").write_bytes(b"broken")
    manifest = pd.DataFrame({"path": ["a/a_1.avi", "a/a_2.avi", "a/a_3.avi"]})

    updated, pairs, _ = assign_groups(manifest, tmp_path, threshold=6)

    assert updated["group_id"].tolist() == ["g0000", "g0000", "g0001"]
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
