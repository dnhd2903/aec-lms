"""
eval/metrics.py — Evaluation metrics of an AEC run (PROJECT_PLAN §6).

    ERLE(n)          = 10·log10( Σ r² / Σ (r − ŷ)² )      over a sliding window, true echo r
    Misalignment(n)  = 20·log10( ‖h − w(n)‖ / ‖h‖ )        how well w identifies the echo path h
    Convergence time = first time the misalignment is within 3 dB of its steady-state value

Both ERLE and misalignment need r and h, which are known only because the echo is
simulated. A real system would use Σ d² / Σ e² on far-end-only segments instead,
which is capped by the echo-to-noise ratio.
"""

import numpy as np


def db(p):
    """Power ratio → dB, with a floor so that zeros do not give −inf."""
    return 10 * np.log10(np.maximum(p, 1e-20))


def sliding_erle_db(r, y_hat, win=400):
    """
    ERLE(n) over a sliding window of win samples (400 = 50 ms at 8 kHz), using the true echo r.
    NaN where there is (almost) no echo, since the ratio is meaningless there.
    """
    k = np.ones(win)
    echo = np.convolve(r ** 2, k, "same")
    erle = db(echo) - db(np.convolve((r - y_hat) ** 2, k, "same"))
    erle[echo < 1e-4 * np.max(echo)] = np.nan
    return erle


def segment_erle_db(r, y_hat, a=0, b=None):
    """One ERLE value over the samples a … b−1."""
    return db(np.sum(r[a:b] ** 2)) - db(np.sum((r[a:b] - y_hat[a:b]) ** 2))


def misalignment_db(h, w_hist):
    """
    Normalised misalignment of each snapshot in w_hist, shape (K, M) → (K,).
    When M ≠ L_h the shorter one is zero-padded.
    """
    w_hist = np.atleast_2d(w_hist)
    L = max(len(h), w_hist.shape[1])
    h = np.pad(h, (0, L - len(h)))
    w_hist = np.pad(w_hist, ((0, 0), (0, L - w_hist.shape[1])))
    return 20 * np.log10(np.linalg.norm(w_hist - h, axis=1) / np.linalg.norm(h))


def steady_state_db(curve_db, tail=0.2):
    """Steady-state level of a dB curve: mean power over its last `tail` fraction, in dB."""
    curve_db = np.asarray(curve_db)
    last = curve_db[-max(1, int(tail * len(curve_db))):]
    return float(db(np.mean(10 ** (last / 10))))


def convergence_index(curve_db, margin_db=3.0, tail=0.2):
    """
    Index of the first point where curve_db is within margin_db of its steady state.
    Multiply by the snapshot spacing (hist_every) to get samples. None if it never gets there.
    """
    target = steady_state_db(curve_db, tail) + margin_db
    hits = np.flatnonzero(np.asarray(curve_db) <= target)
    return int(hits[0]) if len(hits) else None
