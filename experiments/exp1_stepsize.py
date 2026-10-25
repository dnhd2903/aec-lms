"""
experiments/exp1_stepsize.py — Experiment 1: LMS step size, speed versus misadjustment (PROJECT_PLAN §3 point 5).

White-noise far-end, M = L_h = 256, ENR = 35 dB. LMS is run with several step sizes,
written as  α = µ·M·σ²ₓ  (so µ = α / (M·σ²ₓ); the demo uses α = 0.1).

Theory (white Gaussian input, independence assumption) — the dashed curves:

    MSD(n) = E‖h − w(n)‖²  obeys  MSD(n+1) = ρ·MSD(n) + µ²·M·σ²ₓ·σ²ᵥ
    ρ      = 1 − µσ²ₓ·(2 − µσ²ₓ(M+2))                  per-sample decay factor
    MSD(∞) = µ·M·σ²ᵥ / (2 − µσ²ₓ(M+2))                  steady state
    stable ⇔ 0 < µ < 2 / (σ²ₓ(M+2))  ≈  2 / (M·σ²ₓ)

Larger µ ⇒ ρ further below 1 (faster) but larger MSD(∞) (worse steady state), until
α ≈ 1 where speed peaks, then both get worse and α ≥ 2 diverges. For white input the
ERLE is the mirror image of the misalignment (ERLE ≈ −misalignment), so it is only
reported in the table.

    python3 -m experiments.exp1_stepsize        → results/exp1/stepsize.png + a table
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aec.lms import lms
from eval.metrics import convergence_index, misalignment_db, segment_erle_db, steady_state_db
from eval.plots import COLOR_LMS, COLOR_THEORY, save, shades, style
from sim.rir import make_rir
from sim.scenario import build_scenario, power

FS = 8000
M = 256
DURATION = 6.0              # s — long enough for the smallest µ to settle
ALPHAS = [0.05, 0.1, 0.5, 1.0]
ERL, ENR = 8.0, 35.0
HIST_EVERY = 100
OUT = Path("results/exp1")


def lms_theory_curve(mu, sigma2_x, sigma2_v, h_norm2, M, n):
    """Theoretical normalised misalignment (dB) at sample indices n, starting from w = 0."""
    a = mu * sigma2_x
    rho = 1 - a * (2 - a * (M + 2))
    msd_inf = mu * M * sigma2_v / (2 - a * (M + 2))
    msd = (h_norm2 - msd_inf) * rho ** n + msd_inf
    return 10 * np.log10(msd / h_norm2)


def white_noise_scenario(seed=0, duration=DURATION, gain=1.0):
    """Unit-power white-noise far-end (times gain) through a 256-tap RIR, with noise at ENR."""
    x = gain * np.random.default_rng(seed).standard_normal(int(duration * FS))
    h = make_rir(M, 0.1, FS, seed=seed)
    return build_scenario(x, h, erl_db=ERL, enr_db=ENR, seed=seed)


def main():
    x, r, s, v, d, h = white_noise_scenario()
    sigma2_x, sigma2_v = power(x), power(v)
    n_hist = np.arange(len(x) // HIST_EVERY + 1) * HIST_EVERY
    t_hist = n_hist / FS

    print(f"White-noise far-end, {DURATION:.0f} s, M = L_h = {M}, ERL = {ERL:.0f} dB, ENR = {ENR:.0f} dB\n")
    print(f"  {'α = µMσ²':>9s} {'µ':>9s} │ {'t_conv meas.':>12s} {'theory':>7s} │ "
          f"{'steady misal. meas.':>19s} {'theory':>8s} │ {'ERLE (last 1 s)':>15s}")

    fig, (ax_curve, ax_trade) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    points = []
    for alpha, color in zip(ALPHAS, shades(COLOR_LMS, len(ALPHAS))):
        mu = alpha / (M * sigma2_x)
        e, y_hat, w_hist = lms(x, d, M, mu, hist_every=HIST_EVERY)
        mis = misalignment_db(h, w_hist)
        theory = lms_theory_curve(mu, sigma2_x, sigma2_v, h @ h, M, n_hist)

        k_meas, k_theo = convergence_index(mis), convergence_index(theory)
        ss_meas, ss_theo = steady_state_db(mis), steady_state_db(theory)
        t_meas, t_theo = k_meas * HIST_EVERY / FS, k_theo * HIST_EVERY / FS
        erle = segment_erle_db(r, y_hat, len(x) - FS)
        points.append((t_meas, ss_meas, t_theo, ss_theo, alpha, color))
        print(f"  {alpha:9.2f} {mu:9.2e} │ {t_meas:10.2f} s {t_theo:5.2f} s │ "
              f"{ss_meas:16.1f} dB {ss_theo:5.1f} dB │ {erle:12.1f} dB")

        ax_curve.plot(t_hist, mis, color=color, lw=1.8, label=f"α = {alpha:g}")
        ax_curve.plot(t_hist, theory, color=COLOR_THEORY, lw=0.8, ls="--")

    ax_curve.plot([], [], color=COLOR_THEORY, lw=0.8, ls="--", label="theory")
    ax_curve.legend(frameon=False, ncol=2)
    style(ax_curve, "(a) LMS misalignment for µ = α / (M·σ²ₓ)", "Time (s)", "Misalignment (dB)")

    for t_meas, ss_meas, t_theo, ss_theo, alpha, color in points:
        ax_trade.plot(t_meas, ss_meas, "o", color=color, ms=8)
        ax_trade.plot(t_theo, ss_theo, "x", color=COLOR_THEORY, ms=7)
        ax_trade.annotate(f"α = {alpha:g}", (t_meas, ss_meas), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax_trade.plot([], [], "o", color=COLOR_LMS, label="measured")
    ax_trade.plot([], [], "x", color=COLOR_THEORY, label="theory")
    ax_trade.legend(frameon=False)
    style(ax_trade, "(b) Trade-off: faster ⇔ higher floor", "Convergence time to steady state + 3 dB (s)",
          "Steady-state misalignment (dB)")

    fig.suptitle(f"Experiment 1 — LMS step size, white-noise far-end, M = L_h = {M}, ENR = {ENR:.0f} dB", fontsize=11)
    save(fig, OUT / "stepsize.png")


if __name__ == "__main__":
    main()
