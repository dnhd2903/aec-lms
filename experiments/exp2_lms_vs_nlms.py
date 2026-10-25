"""
experiments/exp2_lms_vs_nlms.py — Experiment 2: the two limits of LMS (PROJECT_PLAN §3 points 2–4).

2a  Gain sweep (white noise). µ is fixed once for a far-end at unit power, α₀ = µ·M·σ²ₓ = 0.1,
    then the far-end is played g times louder (the echo and the noise scale with it,
    so ERL and ENR are unchanged — only the input level changes). For LMS the
    effective step is α = g²·α₀, so the same µ converges faster, settles higher, and
    diverges once  g²·α₀·(M+2)/M ≥ 2  (exp1 theory). NLMS divides by ‖x(n)‖², which
    scales with g² too, so its curve does not move at all.

2b  White noise versus speech at the same power, same µ. The LMS speed depends on the
    eigenvalue spread χ = λ_max/λ_min of R_xx = E[x(n)xᵀ(n)]: white noise has χ = 1,
    speech is strongly coloured so χ is huge and the modes with small λ barely move.
    NLMS fixes the level problem (2a), not this one.

    python3 -m experiments.exp2_lms_vs_nlms     → results/exp2/gain.png, results/exp2/speech.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

from aec.lms import lms
from eval.metrics import db, misalignment_db, sliding_erle_db, steady_state_db
from eval.plots import COLOR_LMS, COLOR_NLMS, COLOR_THEORY, save, shades, style
from experiments.demo import load_recording
from experiments.exp1_stepsize import ENR, ERL, FS, HIST_EVERY, M, white_noise_scenario
from sim.rir import make_rir
from sim.scenario import build_scenario, power

ALPHA0 = 0.1                    # LMS step at unit far-end power, as in the demo
MU_NLMS = 0.5
GAINS_SHOWN = [1, 3, 5]         # 0, +9.5, +14 dB
GAINS_SWEEP_DB = np.arange(0, 17)
CEILING_DB = 20                 # diverged runs are drawn at this level
OUT = Path("results/exp2")


def misalignment_or_inf_db(h, w_hist):
    """misalignment_db, with +inf wherever the filter has blown up (overflow or above the ceiling)."""
    with np.errstate(all="ignore"):
        mis = misalignment_db(h, w_hist)
    mis[~np.isfinite(mis) | (mis > CEILING_DB)] = np.inf
    return mis


def final_misalignment_db(h, w_hist):
    """Steady-state misalignment, or +inf if the filter diverged."""
    mis = misalignment_or_inf_db(h, w_hist)
    return np.inf if np.isinf(mis[-1]) else steady_state_db(mis)


def run_pair(x, d, mu_lms, delta=1e-6):
    """LMS and NLMS on the same signals; overflow of a diverging LMS is expected and silenced."""
    with np.errstate(all="ignore"):
        return {
            "LMS": lms(x, d, M, mu_lms, hist_every=HIST_EVERY),
            "NLMS": lms(x, d, M, MU_NLMS, normalize=True, delta=delta, hist_every=HIST_EVERY),
        }


def exp2a():
    mu = ALPHA0 / M                     # σ²ₓ = 1 at g = 1
    g_max_db = 10 * np.log10(2 * M / (ALPHA0 * (M + 2)))
    print(f"2a — gain sweep, white noise, LMS µ = {mu:.2e} (α₀ = {ALPHA0} at g = 1), NLMS µ̃ = {MU_NLMS}")
    print(f"     theory: LMS diverges above g = {g_max_db:.1f} dB (×{10 ** (g_max_db / 20):.2f})\n")
    print(f"  {'gain':>12s} {'α = g²α₀':>9s} │ {'LMS':>9s} {'NLMS':>9s}   (steady-state misalignment)")

    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    sweep = {"LMS": [], "NLMS": []}
    for g_db in GAINS_SWEEP_DB:
        x, r, s, v, d, h = white_noise_scenario(duration=4.0, gain=10 ** (g_db / 20))
        runs = run_pair(x, d, mu)
        for name, (_, _, w_hist) in runs.items():
            sweep[name].append(final_misalignment_db(h, w_hist))
        cells = "".join(f"{m:6.1f} dB" if np.isfinite(m) else f"{'diverged':>9s}" for m in (sweep["LMS"][-1], sweep["NLMS"][-1]))
        print(f"  {g_db:3d} dB (×{10 ** (g_db / 20):4.2f}) {ALPHA0 * 10 ** (g_db / 10):9.2f} │ {cells}")

    t = None
    for g, c_lms, c_nlms in zip(GAINS_SHOWN, shades(COLOR_LMS, 3), shades(COLOR_NLMS, 3)):
        x, r, s, v, d, h = white_noise_scenario(duration=4.0, gain=g)
        runs = run_pair(x, d, mu)
        t = np.arange(len(runs["LMS"][2])) * HIST_EVERY / FS
        mis_lms = misalignment_or_inf_db(h, runs["LMS"][2])
        ax_t.plot(t, np.minimum(mis_lms, CEILING_DB), color=c_lms, lw=1.8, label=f"LMS, ×{g}")
        ax_t.plot(t, misalignment_db(h, runs["NLMS"][2]), color=c_nlms, lw=1.2, ls="--",
                  label="NLMS, ×1 ×3 ×5 (identical)" if g == GAINS_SHOWN[-1] else None)
    ax_t.set_ylim(-55, CEILING_DB + 3)
    ax_t.legend(frameon=False, ncol=2, fontsize=9, loc="center right")
    style(ax_t, "(a) Misalignment, far-end ×1, ×3, ×5 (same µ, same µ̃)", "Time (s)", "Misalignment (dB)")

    for name, color in [("LMS", COLOR_LMS), ("NLMS", COLOR_NLMS)]:
        ax_g.plot(GAINS_SWEEP_DB, np.minimum(sweep[name], CEILING_DB), "o-", color=color, ms=4, label=name)
    ax_g.axvline(g_max_db, color=COLOR_THEORY, ls="--", lw=1)
    ax_g.annotate("LMS stability limit\n(theory)", (g_max_db, -50), xytext=(-8, 0), textcoords="offset points",
                  ha="right", fontsize=9)
    ax_g.set_ylim(-55, CEILING_DB + 3)
    ax_g.legend(frameon=False, loc="upper left")
    style(ax_g, f"(b) Steady state vs far-end gain (diverged ⇒ {CEILING_DB} dB)", "Far-end gain g (dB)",
          "Steady-state misalignment (dB)")

    fig.suptitle("Experiment 2a — LMS stability depends on the input level, NLMS does not", fontsize=11)
    save(fig, OUT / "gain.png")


def eigenvalues_db(x):
    """Eigenvalues of the M×M autocorrelation matrix R_xx estimated from x, in dB, descending."""
    acf = np.correlate(x, x, "full")[len(x) - 1: len(x) - 1 + M] / len(x)
    lam = np.linalg.eigvalsh(toeplitz(acf))[::-1]
    return db(np.maximum(lam, 1e-12 * lam[0]))


def exp2b():
    speech = load_recording(Path("data/far_end.wav"), "far-end")
    speech /= np.sqrt(power(speech))                    # unit power, like the white noise
    white = np.random.default_rng(0).standard_normal(len(speech))
    mu = ALPHA0 / M
    delta = 1e-2 * M                                    # as in the demo, so speech pauses are safe
    print(f"\n2b — white noise vs speech, both unit power, {len(speech) / FS:.0f} s, "
          f"LMS µ = {mu:.2e}, NLMS µ̃ = {MU_NLMS}\n")
    print(f"  {'input':8s} {'spread χ':>10s} │ {'LMS':>9s} {'NLMS':>9s}   (misalignment, last 20 %)")

    h0 = make_rir(M, 0.1, FS, seed=0)
    fig, (ax_eig, ax_mis, ax_erle) = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [0.8, 1.2, 1.2]})
    for label, x0, ls in [("white", white, "-"), ("speech", speech, "--")]:
        x, r, s, v, d, h = build_scenario(x0, h0, erl_db=ERL, enr_db=ENR, seed=0)
        lam = eigenvalues_db(x)
        runs = run_pair(x, d, mu, delta)
        mis = {name: misalignment_db(h, w_hist) for name, (_, _, w_hist) in runs.items()}
        print(f"  {label:8s} {lam[0] - lam[-1]:7.1f} dB │" + "".join(f"{steady_state_db(m):6.1f} dB" for m in mis.values()))

        ax_eig.plot(np.arange(1, M + 1), lam - lam[0], color=COLOR_THEORY, ls=ls, lw=1.5,
                    label=f"{label}, χ = {lam[0] - lam[-1]:.0f} dB")
        t_w = np.arange(len(mis["LMS"])) * HIST_EVERY / FS
        t = np.arange(len(x)) / FS
        for name, color in [("LMS", COLOR_LMS), ("NLMS", COLOR_NLMS)]:
            ax_mis.plot(t_w, mis[name], color=color, ls=ls, lw=1.5, label=f"{name}, {label}")
            if label == "speech":
                ax_erle.plot(t, np.clip(sliding_erle_db(r, runs[name][1]), -10, 60), color=color, lw=0.8, label=name)

    ax_eig.legend(frameon=False, fontsize=9)
    style(ax_eig, "(a) Eigenvalues of R_xx (sorted)", "Index", "λ / λ_max (dB)")
    ax_mis.legend(frameon=False, fontsize=9, ncol=2)
    style(ax_mis, "(b) Misalignment, same power, same µ", "Time (s)", "Misalignment (dB)")
    ax_erle.set_ylim(-10, 60)
    ax_erle.legend(frameon=False, fontsize=9)
    style(ax_erle, "(c) ERLE(t) on speech, 50 ms window", "Time (s)", "ERLE (dB)")
    fig.suptitle(f"Experiment 2b — white noise vs speech at the same power, M = L_h = {M}, ENR = {ENR:.0f} dB",
                 fontsize=11)
    save(fig, OUT / "speech.png")


def main():
    exp2a()
    exp2b()


if __name__ == "__main__":
    main()
