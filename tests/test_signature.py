import numpy as np
import pytest

from flake_scanner.calibration.signature import MaterialSignature


def _sig(n=10):
    rng = np.random.default_rng(0)
    samples = rng.normal([0.44, -0.245, -0.20], 0.03, size=(n, 3))
    return MaterialSignature.from_ndi(samples, "hBN", "SiO2-285nm", 35, 45)


def test_requires_min_samples():
    with pytest.raises(ValueError):
        MaterialSignature.from_ndi(np.zeros((2, 3)), "hBN", "s", 35, 45)


def test_mahalanobis_zero_at_mean():
    sig = _sig()
    assert sig.mahalanobis(sig.mean) == pytest.approx(0.0, abs=1e-9)


def test_mahalanobis_grows_with_distance():
    sig = _sig()
    near = sig.mahalanobis(sig.mean + [0.01, 0.01, 0.01])
    far = sig.mahalanobis(sig.mean + [0.3, 0.3, 0.3])
    assert far > near


def test_field_matches_scalar():
    sig = _sig()
    v = np.array([0.4, -0.2, -0.18])
    field = sig.mahalanobis_field(v.reshape(1, 1, 3))
    assert field[0, 0] == pytest.approx(sig.mahalanobis(v))
