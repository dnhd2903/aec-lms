"""
aec/lms.py — LMS adaptive filter, with NLMS as the normalize=True variant.

One sample per iteration, with the tap-input vector in time-reversed order:

    x(n) = [x(n), x(n−1), …, x(n−M+1)]ᵀ
    y(n) = wᵀ(n) x(n)                       echo estimate
    e(n) = d(n) − y(n)                      error = output of the AEC
    LMS : w(n+1) = w(n) + µ · e(n) · x(n)
    NLMS: w(n+1) = w(n) + µ · e(n) · x(n) / (δ + ‖x(n)‖²)

LMS and NLMS differ only in the step size, so they share one implementation:
the comparison in Experiment 2 runs exactly the same code path.

Stability: LMS needs 0 < µ < 2/λ_max ≈ 2/(M·σ²ₓ), which depends on the input power.
NLMS needs 0 < µ < 2 regardless of the input level. A safe LMS starting point is
µ = 0.1 / (M·σ²ₓ).
"""

from typing import Optional

import numpy as np


def lms(
    x: np.ndarray,
    d: np.ndarray,
    M: int,
    mu: float,
    normalize: bool = False,
    delta: float = 1e-6,
    adapt: Optional[np.ndarray] = None,
    hist_every: int = 100,
):
    """
    Run an LMS (or NLMS) adaptive filter over the whole signal.

    Parameters
    ----------
    x          : far-end signal (filter input), shape (N,).
    d          : microphone signal (desired signal), shape (N,).
    M          : number of filter taps.
    mu         : step size µ (LMS) or normalised step size µ̃ (NLMS, 0 < µ̃ < 2).
    normalize  : False ⇒ LMS, True ⇒ NLMS.
    delta      : regularisation δ of the NLMS denominator, avoids division by ~0
                 when the input is silent. Ignored for LMS.
    adapt      : optional bool mask, shape (N,). Where it is False the filter still
                 filters (y and e are computed) but w is not updated. Used to freeze
                 adaptation during double-talk.
    hist_every : record the coefficients every this many samples.

    Returns
    -------
    e      : error signal d − y, shape (N,) — the AEC output.
    y_hat  : echo estimate, shape (N,).
    w_hist : coefficient snapshots, shape (N // hist_every + 1, M).
             w_hist[k] = w(k·hist_every), the filter after k·hist_every updates,
             so w_hist[0] = 0 and, when N is a multiple of hist_every, w_hist[-1]
             is the final filter.
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    N = len(x)
    if d.shape != x.shape:
        raise ValueError(f"x and d must have the same shape, got {x.shape} vs {d.shape}")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    if adapt is not None:
        adapt = np.asarray(adapt, dtype=bool)
        if adapt.shape != x.shape:
            raise ValueError(f"adapt must have the same shape as x, got {adapt.shape} vs {x.shape}")

    # Zero-pad the past so x_pad[n : n+M] holds x(n−M+1) … x(n); reversing it
    # gives the tap-input vector [x(n), x(n−1), …, x(n−M+1)].
    x_pad = np.concatenate([np.zeros(M - 1), x])

    w = np.zeros(M)
    e = np.empty(N)
    y_hat = np.empty(N)
    w_hist = np.empty((N // hist_every + 1, M))

    for n in range(N):
        if n % hist_every == 0:
            w_hist[n // hist_every] = w

        x_vec = x_pad[n : n + M][::-1]
        y_hat[n] = w @ x_vec
        e[n] = d[n] - y_hat[n]

        if adapt is None or adapt[n]:
            if normalize:
                w += (mu / (delta + x_vec @ x_vec)) * e[n] * x_vec
            else:
                w += mu * e[n] * x_vec

    if N % hist_every == 0:
        w_hist[-1] = w

    return e, y_hat, w_hist
