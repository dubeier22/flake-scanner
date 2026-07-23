import numpy as np

from flake_scanner.calibration.sampling import (
    normalized_difference,
    sample_patch,
    snap_sample,
)


def test_ndi_basic_sign():
    # flake brighter than substrate in a channel -> negative ndi there
    ndi = normalized_difference([200, 100, 50], [100, 100, 100])
    assert ndi[0] < 0  # flake B (200) > sub B (100)
    assert abs(ndi[1]) < 1e-9  # equal
    assert ndi[2] > 0  # flake R (50) < sub R (100)


def test_ndi_multiplicative_invariance():
    # exposure / white-balance = per-channel scaling -> ndi unchanged
    flake = np.array([60.0, 240.0, 180.0])
    sub = np.array([160.0, 130.0, 120.0])
    base = normalized_difference(flake, sub)
    for gain in ([1.3, 1.3, 1.3], [1.0, 1.15, 0.9]):
        g = np.array(gain)
        scaled = normalized_difference(flake * g, sub * g)
        assert np.allclose(base, scaled, atol=1e-9)


def test_ndi_bounded():
    ndi = normalized_difference([0, 0, 0], [255, 255, 255])
    assert np.all(ndi <= 1.0) and np.all(ndi >= -1.0)


def test_sample_patch_clips_at_border():
    img = np.ones((10, 10, 3), np.uint8) * 50
    # near a corner: must not raise and returns the mean colour
    assert np.allclose(sample_patch(img, 0, 0, r=8), 50)


def test_snap_prefers_blob_at_mark_over_bigger_neighbor():
    # substrate grey, a small flake at the mark, a bigger flake to the side
    img = np.full((300, 300, 3), 120, np.uint8)
    img[140:160, 140:160] = (60, 240, 180)  # small flake at mark (green-ish)
    img[100:200, 200:280] = (240, 60, 60)  # bigger flake to the right (blue-ish)
    res = snap_sample(img, 150, 150, window=170)
    assert res is not None
    flake, _sub = res
    # should sample the small green flake at the mark, not the big blue one
    assert flake[1] > flake[0] and flake[1] > flake[2]
