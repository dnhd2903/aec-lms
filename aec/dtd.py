"""
aec/dtd.py — Geigel double-talk detector (PROJECT_PLAN §7, Week 3).

The echo of the far-end is weaker than the far-end itself (ERL ≥ 6 dB ⇒ |r| < 0.5·|x|).
So if the microphone is louder than half the recent far-end peak, someone else must
be talking:

    double-talk(n)  ⇔  |d(n)| > T · max(|x(n)|, |x(n−1)|, …, |x(n−L+1)|),     T ≈ 0.5

L should cover the echo path (L ≥ L_h), because the echo at time n comes from x up
to L_h samples earlier. Once triggered, the flag is held for `hangover` more samples
(30 ms = 240 at 8 kHz), otherwise it would flicker on and off between syllables.

Use it to freeze the adaptation:  lms(x, d, M, mu, adapt=~geigel(x, d)).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def geigel(x: np.ndarray, d: np.ndarray, L: int = 256, T: float = 0.5, hangover: int = 240) -> np.ndarray:
    """
    Parameters
    ----------
    x        : far-end signal, shape (N,).
    d        : microphone signal, shape (N,).
    L        : length of the far-end window whose peak is compared with |d(n)|.
    T        : threshold; 0.5 assumes the echo is at least 6 dB below the far-end.
    hangover : number of samples the flag stays on after the last detection.

    Returns
    -------
    Bool mask, shape (N,): True where double-talk is declared (adaptation should stop).
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if d.shape != x.shape:
        raise ValueError(f"x and d must have the same shape, got {x.shape} vs {d.shape}")

    # Same zero-padding trick as aec/lms.py: window n covers x(n−L+1) … x(n).
    x_pad = np.concatenate([np.zeros(L - 1), np.abs(x)])
    x_peak = sliding_window_view(x_pad, L).max(axis=1)
    detected = np.abs(d) > T * x_peak

    # Hangover: n is flagged if any of the samples n−hangover … n was detected.
    return np.convolve(detected, np.ones(hangover + 1))[: len(x)] > 0
