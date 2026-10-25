"""
experiments/exp3_doubletalk.py — Experiment 3: double-talk, without and with a detector.

Real speech, mode 1 of the demo: far-end data/far_end.wav for 30 s, near-end
data/near_end.wav from 12 s (double-talk 12–22 s), SER = 0 dB.

While both talk, e(n) = (r − y)(n) + s(n) + v(n): the near-end speech s looks like a
huge "error" to the LMS, which then pushes w away from h. Three ways to handle it,
each run with LMS and NLMS:

    no DTD   adapt everywhere (what the demo does)          → the filter diverges
    Geigel   freeze while aec/dtd.py declares double-talk    → the practical fix
    ideal    freeze exactly while the near-end talks         → upper bound of any DTD
             (needs the true s, so only possible in simulation)

    python3 -m experiments.exp3_doubletalk      → results/exp3/doubletalk.png + WAVs + a table
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aec.dtd import geigel
from aec.lms import lms
from eval.metrics import misalignment_db, segment_erle_db, sliding_erle_db
from eval.plots import COLOR_DT, COLOR_LMS, COLOR_NLMS, COLOR_SIGNAL, save, style
from experiments.demo import load_recording
from sim.audio import save_wav
from sim.rir import make_rir
from sim.scenario import build_scenario, power

FS = 8000
M = 256
NEAR_START = 12.0               # s
SER, ERL, ENR = 0.0, 8.0, 35.0
MU_NLMS = 0.5
GEIGEL = dict(L=M, T=0.5, hangover=240)
HIST_EVERY = 100
ERLE_WIN = 4000                 # 0.5 s: smoother than the demo's 50 ms, so six curves stay readable
OUT = Path("results/exp3")
LINESTYLES = {"no DTD": ":", "Geigel": "-", "ideal": "--"}


def active(sig, win=400, rel=1e-2):
    """Where a talker is active: 50 ms power above rel × the mean power of its non-zero part."""
    p = np.convolve(sig ** 2, np.ones(win) / win, "same")
    return p > rel * power(sig[sig != 0])


def main():
    print("Input recordings (after resampling to 8 kHz):")
    x = load_recording(Path("data/far_end.wav"), "far-end")
    near = load_recording(Path("data/near_end.wav"), "near-end")
    N = len(x)
    a = int(NEAR_START * FS)
    b = min(N, a + len(near))
    s = np.zeros(N)
    s[a:b] = near[: b - a]

    h = make_rir(M, 0.1, FS, seed=0)
    x, r, _, v, _, h = build_scenario(x, h, erl_db=ERL, enr_db=ENR, seed=0)
    s *= np.sqrt(power(r) / power(s[a:b]) * 10 ** (SER / 10))
    d = r + s + v

    ideal = np.zeros(N, dtype=bool)
    ideal[a:b] = True
    masks = {"no DTD": np.zeros(N, dtype=bool), "Geigel": geigel(x, d, **GEIGEL), "ideal": ideal}

    far_only = ~ideal
    near_active = active(s) & active(x)
    g = masks["Geigel"]
    print(f"\nGeigel DTD (L = {GEIGEL['L']}, T = {GEIGEL['T']}, hangover = {GEIGEL['hangover']}):")
    print(f"  false alarms  (frozen while only the far-end talks) : {100 * g[far_only].mean():5.1f} %")
    print(f"  detections    (frozen while the near-end is active) : {100 * g[near_active].mean():5.1f} %")

    p_x = power(x)
    mu_lms = 0.1 / (M * p_x)
    delta = 1e-2 * M * p_x
    runs = {}
    for dtd, mask in masks.items():
        runs["LMS", dtd] = lms(x, d, M, mu_lms, adapt=~mask, hist_every=HIST_EVERY)
        runs["NLMS", dtd] = lms(x, d, M, MU_NLMS, normalize=True, delta=delta, adapt=~mask, hist_every=HIST_EVERY)

    print(f"\nLMS µ = {mu_lms:.3g}, NLMS µ̃ = {MU_NLMS}, double-talk {a / FS:.0f}–{b / FS:.0f} s, SER = {SER:.0f} dB")
    print(f"  {'':12s} │ {'ERLE: far-end only':>18s} {'double-talk':>11s} {'after':>7s} │ "
          f"{'misalignment: DT start':>22s} {'DT end':>7s} {'final':>7s}")
    for (alg, dtd), (e, y_hat, w_hist) in runs.items():
        erle = [segment_erle_db(r, y_hat, *seg) for seg in [(0, a), (a, b), (b, N)]]
        mis = misalignment_db(h, w_hist[[a // HIST_EVERY, b // HIST_EVERY, -1]])
        print(f"  {alg:4s} {dtd:7s} │ {erle[0]:15.1f} dB {erle[1]:8.1f} dB {erle[2]:4.1f} dB │ "
              f"{mis[0]:19.1f} dB {mis[1]:4.1f} dB {mis[2]:4.1f} dB")

    # --- audio: one common scale factor, as in the demo ------------------------
    signals = {"far_end_hears_no_aec": d}
    for (alg, dtd), (e, _, _) in runs.items():
        signals[f"far_end_hears_{alg.lower()}_{dtd.lower().replace(' ', '_')}"] = e
    scale = 0.9 / max(np.max(np.abs(sig)) for sig in signals.values())
    for name, sig in signals.items():
        save_wav(OUT / f"{name}.wav", scale * sig, FS)
    print(f"\nSaved {len(signals)} WAVs in {OUT}/")

    # --- figure ----------------------------------------------------------------
    t = np.arange(N) / FS
    t_w = np.arange(len(runs["LMS", "no DTD"][2])) * HIST_EVERY / FS
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1.4, 1.2, 1.2]})

    peak = np.max(np.abs(d))
    axes[0].plot(t, d, color=COLOR_SIGNAL, lw=0.4)
    axes[0].fill_between(t, -peak, -peak + 0.15 * peak, where=g, color=COLOR_NLMS, lw=0, step="mid",
                         label="Geigel: adaptation frozen")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    style(axes[0], "(a) Microphone signal d(n); shaded = double-talk", ylabel="Amplitude")

    for (alg, dtd), (_, y_hat, w_hist) in runs.items():
        color = COLOR_LMS if alg == "LMS" else COLOR_NLMS
        axes[1].plot(t_w, misalignment_db(h, w_hist), color=color, ls=LINESTYLES[dtd], lw=1.6, label=f"{alg}, {dtd}")
        ax = axes[2] if alg == "LMS" else axes[3]
        ax.plot(t, np.clip(sliding_erle_db(r, y_hat, ERLE_WIN), -15, 50), color=color, ls=LINESTYLES[dtd], lw=1.2,
                label=dtd)
    axes[1].legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    style(axes[1], "(b) Normalised misalignment", ylabel="Misalignment (dB)")
    for ax, alg in [(axes[2], "LMS"), (axes[3], "NLMS")]:
        ax.set_ylim(-15, 50)
        ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower left")
        style(ax, f"({'c' if alg == 'LMS' else 'd'}) ERLE(t), {alg}, true echo, {ERLE_WIN / FS:.1f} s window",
              ylabel="ERLE (dB)")
    axes[3].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvspan(a / FS, b / FS, color=COLOR_DT, zorder=0, lw=0)

    fig.suptitle(f"Experiment 3 — double-talk, SER = {SER:.0f} dB, M = L_h = {M}, ERL = {ERL:.0f} dB, ENR = {ENR:.0f} dB",
                 fontsize=11)
    save(fig, OUT / "doubletalk.png")


if __name__ == "__main__":
    main()
