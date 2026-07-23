import numpy as np

from flake_scanner.calibration.sampling import normalized_difference, sample_box


def test_sample_box_isolates_in_box_flake():
    # grey substrate, a green flake inside the box region, a red-ish flake outside
    img = np.full((300, 300, 3), 120, np.uint8)
    img[130:170, 130:170] = (60, 240, 180)  # target flake inside box
    img[40:80, 40:80] = (240, 60, 60)  # distractor flake outside box
    res = sample_box(img, 120, 120, 60, 60)
    assert res is not None
    flake, sub = res
    # substrate ~grey, flake ~green (G highest) — not the red distractor
    assert flake[1] > flake[0] and flake[1] > flake[2]
    assert abs(sub[0] - 120) < 20
    ndi = normalized_difference(flake, sub)
    assert ndi[1] < 0  # flake brighter than substrate in green


def test_sample_box_excludes_ink():
    img = np.full((200, 200, 3), 120, np.uint8)
    img[90:110, 90:110] = (60, 240, 180)  # flake
    ink = np.zeros((200, 200), np.uint8)
    ink[95:100, 95:100] = 1  # some "ink" over the flake
    res = sample_box(img, 80, 80, 40, 40, exclude_mask=ink)
    assert res is not None
