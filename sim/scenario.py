"""
sim/scenario.py — Build the microphone signal of an echo-cancellation scenario.

Signal model (PROJECT_PLAN §2):

    r(n) = (h * x)(n)                 echo of the far-end signal through the echo path
    d(n) = r(n) + s(n) + v(n)         microphone signal = desired signal of the LMS

Levels (PROJECT_PLAN §4) are set by measuring the actual signal powers:

    ERL = 10·log10(P_x / P_r)    echo return loss; h is rescaled so the echo is erl_db below x
    ENR = 10·log10(P_r / P_v)    echo-to-noise ratio; sets the noise floor of any ERLE measured from e

Powers are averages over the whole signal, so for speech (with pauses) they are
long-term averages, not active-speech levels.
"""

from typing import NamedTuple, Optional

import numpy as np

_NOISE_STREAM = 0x5CE7


class Scenario(NamedTuple):
    """All signals of one scenario. Unpacks as  x, r, s, v, d, h = build_scenario(...)."""
    x: np.ndarray   # far-end (loudspeaker) signal — input of the LMS
    r: np.ndarray   # true echo h * x
    s: np.ndarray   # near-end speech (zeros if absent)
    v: np.ndarray   # background noise (zeros if absent)
    d: np.ndarray   # microphone signal r + s + v — desired signal of the LMS
    h: np.ndarray   # true echo path, rescaled to the requested ERL


def power(x: np.ndarray) -> float:
    """Mean power of a signal, E[x²]."""
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def build_scenario(
    x: np.ndarray,
    h: np.ndarray,
    s: Optional[np.ndarray] = None,
    erl_db: float = 8.0,
    enr_db: Optional[float] = 35.0,
    seed: int = 0,
) -> Scenario:
    """
    Pass the far-end signal through the echo path and add near-end speech and noise.

    Parameters
    ----------
    x      : far-end signal, shape (N,).
    h      : echo path (e.g. from sim.rir.make_rir); only its shape matters, its gain
             is overwritten so that the echo r is erl_db below x.
    s      : near-end speech, shape (N,), used as given (the caller sets its level
             and where it is active). None ⇒ no near-end speech.
    erl_db : echo return loss in dB, P_x / P_r. The plan uses 6–10 dB.
    enr_db : echo-to-noise ratio in dB, P_r / P_v. None ⇒ no noise.
    seed   : seed of the noise generator.

    Returns
    -------
    Scenario(x, r, s, v, d, h), all float64 arrays of length N (h of length L_h).
    """
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = len(x)
    if x.ndim != 1 or h.ndim != 1:
        raise ValueError("x and h must be 1-D arrays")
    if power(x) == 0.0:
        raise ValueError("far-end signal x is all zeros")

    # Echo with the shape of h, then rescale h and r together to hit the ERL.
    r = np.convolve(x, h)[:N]
    gain = np.sqrt(power(x) / power(r) / 10.0 ** (erl_db / 10.0))
    h = gain * h
    r = gain * r

    if s is None:
        s = np.zeros(N)
    else:
        s = np.asarray(s, dtype=np.float64)
        if s.shape != x.shape:
            raise ValueError(f"s must have the same shape as x, got {s.shape} vs {x.shape}")

    if enr_db is None:
        v = np.zeros(N)
    else:
        # The extra key gives a stream distinct from default_rng(seed): a caller that
        # generates x with the same seed would otherwise get v = const · x, i.e. noise
        # fully correlated with the input, which the filter would then "cancel".
        rng = np.random.default_rng([seed, _NOISE_STREAM])
        v =np.sqrt(power(r) / 10.0 ** (enr_db / 10.0)) * rng.standard_normal(N)

    d = r + s + v
    return Scenario(x=x, r=r, s=s, v=v, d=d, h=h)
