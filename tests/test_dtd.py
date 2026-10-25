"""
Geigel DTD: quiet on far-end-only signals at the plan's ERL, fires on a loud near-end,
and holds the flag for the hangover.
"""

import numpy as np

from aec.dtd import geigel
from sim.rir import make_rir
from sim.scenario import build_scenario

FS = 8000


def test_rarely_fires_on_far_end_only():
    # With ERL = 8 dB a single sample trips the detector only ~0.05 % of the time,
    # but each false detection is held for the 240-sample hangover, so ~6 % of the
    # samples end up frozen. Above ~10 % the echo level in sim/ would be wrong.
    x = np.random.default_rng(0).standard_normal(4 * FS)
    sc = build_scenario(x, make_rir(256, 0.1, FS, seed=0), erl_db=8.0, enr_db=35.0)
    assert geigel(sc.x, sc.d).mean() < 0.10


def test_fires_on_loud_near_end():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4 * FS)
    s = np.zeros(4 * FS)
    s[2 * FS: 3 * FS] = 2.0 * rng.standard_normal(FS)       # near-end louder than the far-end
    sc = build_scenario(x, make_rir(256, 0.1, FS, seed=1), s=s, erl_db=8.0, enr_db=35.0)
    mask = geigel(sc.x, sc.d)
    assert mask[2 * FS: 3 * FS].mean() > 0.9
    assert mask[: 2 * FS].mean() < 0.10


def test_hangover_holds_the_flag():
    x = np.ones(1000)
    d = np.zeros(1000)
    d[100] = 10.0                                            # one detection at n = 100
    mask = geigel(x, d, L=8, hangover=50)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(100, 151))
