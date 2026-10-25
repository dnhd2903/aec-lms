import numpy as np
import pytest

from sim.rir import make_rir, tail_level_db


def test_shape_and_unit_energy():
    h = make_rir(256, 0.1, 8000, seed=1)
    assert h.shape == (256,)
    assert np.linalg.norm(h) == pytest.approx(1.0)


def test_seed_is_reproducible():
    assert np.array_equal(make_rir(256, 0.1, 8000, seed=3), make_rir(256, 0.1, 8000, seed=3))
    assert not np.array_equal(make_rir(256, 0.1, 8000, seed=3), make_rir(256, 0.1, 8000, seed=4))


def test_decay_matches_rt60():
    # Make the RIR long enough to decay the full 60 dB, then measure RT60 back with
    # Schroeder backward integration: EDC(n) = Σ_{k≥n} h²(k). Fit a line over the
    # −5 … −35 dB range and extrapolate to −60 dB.
    fs, rt60 = 8000, 0.3
    h = make_rir(int(fs * rt60), rt60, fs, seed=0)
    edc = np.cumsum(h[::-1] ** 2)[::-1]
    edc_db = 10 * np.log10(edc / edc[0])
    idx = np.where((edc_db <= -5) & (edc_db >= -35))[0]
    slope_db_per_sample = np.polyfit(idx, edc_db[idx], 1)[0]
    rt60_measured = -60.0 / slope_db_per_sample / fs
    assert rt60_measured == pytest.approx(rt60, rel=0.1)


def test_tail_level():
    assert tail_level_db(801, 0.1, 8000) == pytest.approx(-60.0)


def test_invalid_args():
    with pytest.raises(ValueError):
        make_rir(0, 0.1, 8000)
    with pytest.raises(ValueError):
        make_rir(256, 0.0, 8000)
