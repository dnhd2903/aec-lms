"""
Week 1 gate (PROJECT_PLAN §7): on the easy case — white-noise far-end, no near-end
speech, no noise, M = L_h — both LMS and NLMS must reach a misalignment below −30 dB.
"""

import numpy as np
import pytest

from aec.lms import lms
from sim.rir import make_rir
from sim.scenario import build_scenario, power

FS = 8000
M = 256


def misalignment_db(h, w):
    return 20 * np.log10(np.linalg.norm(h - w) / np.linalg.norm(h))


@pytest.fixture(scope="module")
def easy_case():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3 * FS)
    h = make_rir(M, 0.1, FS, seed=0)
    return build_scenario(x, h, erl_db=8.0, enr_db=None)


@pytest.mark.parametrize(
    "normalize, mu",
    [
        (False, None),   # LMS, µ = 0.1 / (M·σ²ₓ), set from the data below
        (True, 0.5),     # NLMS
    ],
    ids=["LMS", "NLMS"],
)
def test_converges_on_easy_case(easy_case, normalize, mu):
    x, r, s, v, d, h = easy_case
    if mu is None:
        mu = 0.1 / (M * power(x))
    e, y_hat, w_hist = lms(x, d, M, mu, normalize=normalize)
    assert misalignment_db(h, w_hist[-1]) < -30.0
    # Misalignment must decrease over time, not just end low.
    assert misalignment_db(h, w_hist[len(w_hist) // 4]) > misalignment_db(h, w_hist[-1])


def test_tap_order_identifies_a_pure_delay():
    # Echo path = 3-sample delay. With the correct time-reversed tap vector the
    # filter converges to a single spike at tap 3, not at tap M−1−3.
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000)
    h = np.zeros(16)
    h[3] = 1.0
    _, _, w_hist = lms(x, np.convolve(x, h)[: len(x)], 16, mu=0.5, normalize=True)
    assert np.argmax(np.abs(w_hist[-1])) == 3
    assert misalignment_db(h, w_hist[-1]) < -60.0


def test_scenario_levels():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4 * FS)
    sc = build_scenario(x, make_rir(M, 0.1, FS, seed=2), erl_db=8.0, enr_db=35.0, seed=2)
    assert 10 * np.log10(power(sc.x) / power(sc.r)) == pytest.approx(8.0, abs=1e-9)
    assert 10 * np.log10(power(sc.r) / power(sc.v)) == pytest.approx(35.0, abs=0.2)
    np.testing.assert_allclose(sc.r, np.convolve(sc.x, sc.h)[: len(x)])
    np.testing.assert_allclose(sc.d, sc.r + sc.s + sc.v)


def test_noise_is_uncorrelated_with_far_end_using_the_same_seed():
    # x and the scenario share seed 0: v must still be independent of x.
    x = np.random.default_rng(0).standard_normal(4 * FS)
    sc = build_scenario(x, make_rir(M, 0.1, FS, seed=0), enr_db=35.0, seed=0)
    assert abs(np.corrcoef(sc.x, sc.v)[0, 1]) < 0.05


def test_misalignment_floor_with_noise():
    # With ENR = 35 dB the LMS steady-state misalignment ≈ µ·M·σ²ₓ/2 · P_v/P_r
    # ≈ 0.05 · 10^−3.5 ≈ −48 dB. A floor stuck at −35 dB means v leaks into x.
    x = np.random.default_rng(0).standard_normal(5 * FS)
    x, r, s, v, d, h = build_scenario(x, make_rir(M, 0.1, FS, seed=0), enr_db=35.0, seed=0)
    _, _, w_hist = lms(x, d, M, 0.1 / (M * power(x)))
    assert misalignment_db(h, w_hist[-1]) < -42.0


def test_adapt_mask_freezes_the_filter():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1000)
    d = np.convolve(x, [0.5, -0.3])[:1000]
    adapt = np.ones(1000, dtype=bool)
    adapt[500:] = False
    e, y_hat, w_hist = lms(x, d, 8, mu=0.5, normalize=True, adapt=adapt)
    np.testing.assert_array_equal(w_hist[5], w_hist[-1])    # w(500) == w(1000)
    np.testing.assert_allclose(e, d - y_hat)                 # still filtering


def test_nlms_is_invariant_to_input_gain():
    # Scaling both x and d by g leaves the NLMS update unchanged (up to δ),
    # whereas the LMS step would scale by g².
    rng = np.random.default_rng(4)
    x = rng.standard_normal(2000)
    d = np.convolve(x, make_rir(32, 0.05, FS, seed=4))[:2000]
    _, _, w1 = lms(x, d, 32, mu=0.5, normalize=True, delta=0.0)
    _, _, w10 = lms(10 * x, 10 * d, 32, mu=0.5, normalize=True, delta=0.0)
    np.testing.assert_allclose(w1, w10, atol=1e-10)


def test_w_hist_shape():
    x = np.random.default_rng(5).standard_normal(1050)
    _, _, w_hist = lms(x, x, 4, mu=0.01, hist_every=100)
    assert w_hist.shape == (11, 4)
    assert not w_hist[0].any()
