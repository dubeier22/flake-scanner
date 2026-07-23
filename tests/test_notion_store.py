import numpy as np

from flake_scanner.calibration.notion import chip_of, date_key
from flake_scanner.calibration.store import CalibrationRecord, CalibrationStore


def test_date_key():
    assert date_key("July 7, 2026") == "2026_07_07"
    assert date_key("June 26, 2026") == "2026_06_26"
    assert date_key("garbage") is None


def test_chip_of():
    assert chip_of("B1") == "B"
    assert chip_of("C14") == "C"
    assert chip_of("") == ""


def _rec(t, fl, sub):
    return CalibrationRecord("hBN", "SiO2-285nm", "B", "B1", t,
                             fl[0], fl[1], fl[2], sub[0], sub[1], sub[2], "img.jpg")


def test_store_roundtrip_and_signature(tmp_path):
    db = tmp_path / "cal.csv"
    store = CalibrationStore(db)
    for _ in range(5):
        store.add(_rec(40, [60, 240, 180], [160, 130, 120]))
    store.save()

    reloaded = CalibrationStore(db)
    assert len(reloaded.records) == 5
    sig = reloaded.signature("hBN", "SiO2-285nm", 35, 45)
    assert sig.n_flakes == 5
    # ndi of the stored flake/sub, recomputed
    expected = (np.array([160, 130, 120]) - np.array([60, 240, 180])) / (
        np.array([160, 130, 120]) + np.array([60, 240, 180])
    )
    assert np.allclose(sig.mean, expected, atol=1e-6)


def test_signature_out_of_band_is_empty(tmp_path):
    store = CalibrationStore(tmp_path / "cal.csv")
    for _ in range(5):
        store.add(_rec(120, [60, 240, 180], [160, 130, 120]))  # thick flakes
    try:
        store.signature("hBN", "SiO2-285nm", 35, 45)
        assert False, "expected ValueError for empty band"
    except ValueError:
        pass
