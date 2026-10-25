"""
sim/rir.py — Synthetic room impulse response (RIR), used as the echo path h.

Model (PROJECT_PLAN §4): white Gaussian noise times an exponential decay envelope

    h(n) = g(n) · 10^(−3n / (fs·RT60)),     g(n) ~ N(0, 1),   n = 0 … L−1

The envelope is chosen so that the *energy* drops by exactly 60 dB after RT60 seconds:
    20·log10(10^(−3)) = −60 dB  at  n = fs·RT60.

The output is normalised to unit energy ‖h‖ = 1. The echo level (ERL) is set later,
in sim/scenario.py, by measuring the actual powers of x and r.

Try it from the project root:
    python3 -m sim.rir                          # plot the RIR → results/rir.png
    python3 -m sim.rir --wav data/far_end.wav   # also export a listening demo with echo
"""

import numpy as np


def make_rir(L: int, rt60: float, fs: int, seed: int = 0) -> np.ndarray:
    """
    Generate a synthetic RIR of L samples.

    Parameters
    ----------
    L    : RIR length in samples (= L_h in the plan).
    rt60 : reverberation time in seconds (time for the energy to drop 60 dB).
    fs   : sample rate in Hz.
    seed : random generator seed — same seed ⇒ same h.

    Returns
    -------
    h : float64 ndarray, shape (L,), with ‖h‖₂ = 1.
    """
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if rt60 <= 0:
        raise ValueError(f"rt60 must be > 0, got {rt60}")

    rng = np.random.default_rng(seed)
    n = np.arange(L)
    envelope = 10.0 ** (-3.0 * n / (fs * rt60))
    h = rng.standard_normal(L) * envelope
    return h / np.linalg.norm(h)


def tail_level_db(L: int, rt60: float, fs: int) -> float:
    """
    Level of the envelope at the last sample relative to the first, in dB.

    Tells you where the RIR is truncated: −60 dB means it has fully decayed,
    −10 dB means the tail is cut off while the reverb is still loud.
    """
    return -60.0 * (L - 1) / (fs * rt60)


# ----------------------------------------------------------------------------
# Demo — runs only when executed directly, never on import
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sim.audio import load_wav, save_wav

    parser = argparse.ArgumentParser(description="Plot a synthetic RIR and optionally listen to the echo it creates.")
    parser.add_argument("--L", type=int, default=256, help="RIR length in samples (default 256)")
    parser.add_argument("--rt60", type=float, default=0.1, help="reverberation time in seconds (default 0.1)")
    parser.add_argument("--fs", type=int, default=8000, help="sample rate in Hz (default 8000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wav", type=Path, help="far-end WAV file to convolve with h")
    args = parser.parse_args()

    results = Path("results")
    results.mkdir(exist_ok=True)

    h = make_rir(args.L, args.rt60, args.fs, args.seed)
    t_ms = np.arange(args.L) / args.fs * 1e3
    env_db = -60.0 * np.arange(args.L) / (args.fs * args.rt60)

    print(f"RIR: L = {args.L} samples ({t_ms[-1]:.1f} ms), RT60 = {args.rt60*1e3:.0f} ms, fs = {args.fs} Hz")
    print(f"||h|| = {np.linalg.norm(h):.4f}, envelope at last sample: {tail_level_db(args.L, args.rt60, args.fs):.1f} dB")

    # --- figure: h(n) over time + |h(n)|² in dB ------------------------------
    # The theoretical envelope (dashed) is offset to the mean of |h|² over the
    # first few samples, so only its slope is being compared.
    ref_db = 10 * np.log10(np.mean(h[:8] ** 2))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t_ms, h, lw=0.8)
    ax1.set_ylabel("h(n)")
    ax1.set_title(f"Synthetic RIR — RT60 = {args.rt60*1e3:.0f} ms, L = {args.L}, fs = {args.fs} Hz")
    ax1.grid(alpha=0.3)
    ax2.plot(t_ms, 10 * np.log10(h ** 2 + 1e-12), lw=0.6, label="|h(n)|² (dB)")
    ax2.plot(t_ms, ref_db + env_db, "--", lw=1.5, label="theoretical envelope")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Level (dB)")
    ax2.set_ylim(ref_db + env_db[-1] - 30, ref_db + 10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(results / "rir.png", dpi=150)
    print(f"Saved {results / 'rir.png'}")

    # --- listening demo: dry far-end x and echo r = h * x --------------------
    if args.wav is not None:
        x = load_wav(args.wav, args.fs)
        r = np.convolve(x, h)[: len(x)]
        # One common scale factor for both files, so their relative loudness is preserved.
        scale = min(1.0, 0.99 / max(np.max(np.abs(x)), np.max(np.abs(r))))
        save_wav(results / "rir_demo_dry.wav", scale * x, args.fs)
        save_wav(results / "rir_demo_echo.wav", scale * r, args.fs)
        print(f"Saved {results / 'rir_demo_dry.wav'} and {results / 'rir_demo_echo.wav'}")
