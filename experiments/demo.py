"""
experiments/demo.py — End-to-end listening demo: real speech in, echo simulated, LMS/NLMS cancel it.

Two modes:

1. Two recordings, one near-end segment (default):

       far-end x :  |██████████████████████████████|
       near-end s:             |██████████|            starts at --near-start seconds
                    converge    double-talk   recover

       python3 -m experiments.demo                          # data/far_end.wav (+ data/near_end.wav if present)
       python3 -m experiments.demo --no-near                # far-end only: the output should go quiet
       python3 -m experiments.demo --near-start 16

2. A scripted two-party call (sim/conversation.py, lines in data/call/script.vi.md):

       python3 -m experiments.demo --call data/call         # data/call/far_end.wav + data/call/near_end.wav

No double-talk detector yet (Week 3): adaptation runs everywhere, so during
double-talk the filter is disturbed by s(n). That is expected and is exactly
what Experiment 3 will show.

Outputs in --out (default results/demo or results/call) — all WAVs share one
scale factor, so their relative loudness is what you would actually hear:
    far_end.wav                 x(n): what the far-end talker says, played by the near-end loudspeaker
    far_end_hears_no_aec.wav    d(n): what the far-end talker receives without AEC (the raw
                                microphone signal: near-end voice + echo of their own voice)
    far_end_hears_lms.wav       e(n): what the far-end talker receives with the LMS AEC
    far_end_hears_nlms.wav      e(n): what the far-end talker receives with the NLMS AEC
    demo.png                    waveforms, ERLE(t) and misalignment(t)

Any far-end voice heard in the far_end_hears_*.wav files is echo.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from aec.lms import lms
from sim.audio import load_wav, save_wav
from sim.conversation import CALL_SCRIPT, place_timeline, split_utterances
from sim.rir import make_rir
from sim.scenario import build_scenario, power

FS = 8000
M = 256                 # default filter length (PROJECT_PLAN §4)
L_H = 256               # default echo-path length
ERLE_WIN = 400          # 50 ms sliding window (PROJECT_PLAN §6)
HIST_EVERY = 100
MIN_SEGMENT = 0.3       # shorter activity segments are not listed in the ERLE table

COLOR_LMS = "#2a78d6"
COLOR_NLMS = "#eb6834"
COLOR_SIGNAL = "#8a8a85"
COLOR_NEAR_ONLY = "#efeee9"
COLOR_DT = "#dcdad2"


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-20))


def load_recording(path, name):
    """Load a recording at FS, remove its DC offset and report its level."""
    if not path.exists():
        raise SystemExit(f"{name} file not found: {path}  (record it first, see the recording guide)")
    x = load_wav(path, FS)
    x -= np.mean(x)                                  # some sound cards add a DC offset
    peak = np.max(np.abs(x))
    rms_db = db(power(x))
    print(f"  {name:10s} {len(x) / FS:5.1f} s   peak {20 * np.log10(max(peak, 1e-10)):6.1f} dBFS   RMS {rms_db:6.1f} dBFS")
    if peak >= 0.99:
        print(f"  ⚠ {name}: peak reaches full scale — the recording is probably clipped, record a bit softer")
    if rms_db < -45:
        print(f"  ⚠ {name}: very quiet — raise the input volume or speak closer to the microphone")
    return x


def spans_to_mask(spans, N):
    mask = np.zeros(N, dtype=bool)
    for a, b in spans:
        mask[a:b] = True
    return mask


def activity_segments(far_mask, near_mask):
    """Split the timeline into contiguous (label, start, end) segments by who is talking."""
    labels = np.array(["silence", "far-end only", "near-end only", "double-talk"])
    code = far_mask.astype(int) + 2 * near_mask.astype(int)
    change = np.flatnonzero(np.diff(code)) + 1
    bounds = np.concatenate([[0], change, [len(code)]])
    return [(labels[code[a]], a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def sliding_erle_db(r, y_hat, win=ERLE_WIN):
    """
    ERLE(n) = 10·log10( Σ r² / Σ (r − ŷ)² ) over a sliding window, using the true echo r.
    NaN where there is (almost) no echo, since the ratio is meaningless there.
    """
    k = np.ones(win)
    echo = np.convolve(r ** 2, k, "same")
    erle = db(echo) - db(np.convolve((r - y_hat) ** 2, k, "same"))
    erle[echo < 1e-4 * np.max(echo)] = np.nan
    return erle


def segment_erle_db(r, y_hat, a, b):
    return db(np.sum(r[a:b] ** 2)) - db(np.sum((r[a:b] - y_hat[a:b]) ** 2))


def misalignment_db(h, w_hist):
    """Normalised misalignment; when M ≠ L_h the shorter one is zero-padded."""
    L = max(len(h), w_hist.shape[1])
    h = np.pad(h, (0, L - len(h)))
    w_hist = np.pad(w_hist, ((0, 0), (0, L - w_hist.shape[1])))
    return 20 * np.log10(np.linalg.norm(w_hist - h, axis=1) / np.linalg.norm(h))


def build_two_recordings(args):
    """Mode 1: one far-end recording, optionally one near-end segment at --near-start."""
    x = load_recording(args.far, "far-end")
    N = len(x)
    s = np.zeros(N)
    near_spans = []
    if not args.no_near and args.near.exists():
        near = load_recording(args.near, "near-end")
        start = int(args.near_start * FS)
        if start >= N:
            raise SystemExit(f"--near-start {args.near_start} s is beyond the far-end length ({N / FS:.1f} s)")
        if start + len(near) > N:
            print(f"  note: near-end cut to {(N - start) / FS:.1f} s to fit inside the far-end signal")
            near = near[: N - start]
        s[start: start + len(near)] = near
        near_spans = [(start, start + len(near))]
    elif not args.no_near:
        print(f"  ({args.near} not found — running far-end only)")
    return x, s, [(0, N)], near_spans


def build_call(call_dir):
    """Mode 2: cut each talker's recording into lines and place them following CALL_SCRIPT."""
    far_utts = split_utterances(load_recording(call_dir / "far_end.wav", "far-end"), FS)
    near_utts = split_utterances(load_recording(call_dir / "near_end.wav", "near-end"), FS)
    for name, utts in [("far-end", far_utts), ("near-end", near_utts)]:
        print(f"  {name} lines found: " + ", ".join(f"{len(u) / FS:.1f} s" for u in utts))
    try:
        call = place_timeline(CALL_SCRIPT, far_utts, near_utts, FS)
    except ValueError as err:
        raise SystemExit(f"Cannot assemble the call: {err}.\n"
                         "Check that each line is followed by ~2 s of silence and that no line is missing.")
    print("  timeline: " + "  ".join(f"{tid} {a / FS:.1f}–{b / FS:.1f}s" for tid, (a, b) in call.spans.items()))
    far_spans = [span for tid, span in call.spans.items() if tid.startswith("A")]
    near_spans = [span for tid, span in call.spans.items() if tid.startswith("B")]
    return call.x, call.s, far_spans, near_spans


def main():
    parser = argparse.ArgumentParser(description="AEC listening demo with LMS and NLMS.")
    parser.add_argument("--far", type=Path, default=Path("data/far_end.wav"), help="far-end speech (loudspeaker)")
    parser.add_argument("--near", type=Path, default=Path("data/near_end.wav"), help="near-end speech (local talker)")
    parser.add_argument("--no-near", action="store_true", help="ignore the near-end file: far-end only")
    parser.add_argument("--near-start", type=float, default=12.0, help="when the near-end talker starts, in seconds")
    parser.add_argument("--call", type=Path, metavar="DIR",
                        help="scripted call mode: DIR holds far_end.wav and near_end.wav, one line after another")
    parser.add_argument("--ser", type=float, default=0.0,
                        help="near-end to echo ratio in dB, each measured while talking (default 0)")
    parser.add_argument("--erl", type=float, default=8.0, help="echo return loss in dB (default 8)")
    parser.add_argument("--enr", type=float, default=35.0, help="echo-to-noise ratio in dB (default 35)")
    parser.add_argument("--rt60", type=float, default=0.1, help="reverberation time of the echo path in s (default 0.1)")
    parser.add_argument("--lh", type=int, default=L_H, help=f"echo-path length L_h in samples (default {L_H} = 32 ms)")
    parser.add_argument("--taps", type=int, help="adaptive filter length M in samples (default = --lh)")
    parser.add_argument("--mu-nlms", type=float, default=0.5, help="NLMS normalised step size (0 < µ̃ < 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="output folder (default results/demo, or results/call with --call)")
    args = parser.parse_args()
    out = args.out or Path("results/call" if args.call else "results/demo")

    # --- far-end track x, near-end track s, and who talks when -----------------
    print("Input recordings (after resampling to 8 kHz):")
    x, s, far_spans, near_spans = build_call(args.call) if args.call else build_two_recordings(args)
    N = len(x)
    far_mask, near_mask = spans_to_mask(far_spans, N), spans_to_mask(near_spans, N)

    # --- scenario: echo, noise, near-end at the requested level --------------
    M = args.taps or args.lh
    h = make_rir(args.lh, args.rt60, FS, seed=args.seed)
    x, r, _, v, _, h = build_scenario(x, h, erl_db=args.erl, enr_db=args.enr, seed=args.seed)
    if near_mask.any():
        s = s * np.sqrt(power(r[far_mask]) / power(s[near_mask]) * 10 ** (args.ser / 10))
    d = r + s + v

    # --- run both algorithms on exactly the same signals ---------------------
    # Step sizes use the far-end power while it talks, not diluted by the pauses.
    p_x = power(x[far_mask])
    mu_lms = 0.1 / (M * p_x)
    # δ scaled to the signal level: during far-end pauses ‖x‖² → 0 and a tiny δ
    # would let the noise v drive huge NLMS updates.
    delta = 1e-2 * M * p_x
    print(f"\nRunning LMS (µ = {mu_lms:.3g}) and NLMS (µ̃ = {args.mu_nlms}, δ = {delta:.3g}) on {N / FS:.1f} s ...")
    runs = {
        "LMS": lms(x, d, M, mu_lms, hist_every=HIST_EVERY),
        "NLMS": lms(x, d, M, args.mu_nlms, normalize=True, delta=delta, hist_every=HIST_EVERY),
    }

    # --- numbers ---------------------------------------------------------------
    segments = activity_segments(far_mask, near_mask)
    print("\nTrue ERLE = 10·log10(Σr² / Σ(r − ŷ)²) per segment, higher is better (good AEC: 20–40 dB).")
    print("No far-end ⇒ no echo ⇒ no ERLE (—).")
    print(f"  {'time (s)':13s} {'who talks':15s}" + "".join(f"{name:>10s}" for name in runs))
    for label, a, b in segments:
        if label == "silence" or (b - a) / FS < MIN_SEGMENT:
            continue
        if label in ("far-end only", "double-talk"):
            cells = "".join(f"{segment_erle_db(r, y, a, b):7.1f} dB" for _, y, _ in runs.values())
        else:
            cells = "".join(f"{'—':>10s}" for _ in runs)
        print(f"  {a / FS:5.1f} – {b / FS:5.1f}  {label:15s}" + cells)
    print(f"  {'final misalignment':29s}" + "".join(f"{misalignment_db(h, w[-1:])[0]:7.1f} dB" for _, _, w in runs.values()))

    # --- audio: one common scale factor so loudness differences are preserved -
    signals = {
        "far_end": x,
        "far_end_hears_no_aec": d,
        "far_end_hears_lms": runs["LMS"][0],
        "far_end_hears_nlms": runs["NLMS"][0],
    }
    scale = 0.9 / max(np.max(np.abs(sig)) for sig in signals.values())
    for name, sig in signals.items():
        save_wav(out / f"{name}.wav", scale * sig, FS)
    print(f"\nSaved in {out}/:")
    for name in signals:
        print(f"  {name}.wav")

    # --- figure ----------------------------------------------------------------
    t = np.arange(N) / FS
    t_w = np.arange(len(runs["LMS"][2])) * HIST_EVERY / FS
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.4, 1.4]})

    for ax, sig, title in [
        (axes[0], d, "Microphone signal d(n)"),
        (axes[1], runs["NLMS"][0], "AEC output e(n), NLMS"),
    ]:
        ax.plot(t, sig, color=COLOR_SIGNAL, lw=0.4)
        ax.set_ylim(-1.05 * np.max(np.abs(d)), 1.05 * np.max(np.abs(d)))
        ax.set_ylabel("Amplitude")
        ax.set_title(title, loc="left", fontsize=10)

    erle_lim = (-10, 60)
    for name, color in [("LMS", COLOR_LMS), ("NLMS", COLOR_NLMS)]:
        e, y_hat, w_hist = runs[name]
        axes[2].plot(t, np.clip(sliding_erle_db(r, y_hat), *erle_lim), color=color, lw=1, label=name)
        axes[3].plot(t_w, misalignment_db(h, w_hist), color=color, lw=2, label=name)
    axes[2].set_ylim(*erle_lim)
    axes[2].set_ylabel("ERLE (dB)")
    axes[2].set_title(f"ERLE(t), true echo, {ERLE_WIN / FS * 1e3:.0f} ms window (blank where there is no echo)",
                      loc="left", fontsize=10)
    axes[3].set_ylabel("Misalignment (dB)")
    axes[3].set_title("Normalised misalignment 20·log10(‖h − w‖ / ‖h‖)", loc="left", fontsize=10)
    axes[3].set_xlabel("Time (s)")
    for ax in axes[2:]:
        ax.legend(loc="upper right", frameon=False)

    shades = {"near-end only": COLOR_NEAR_ONLY, "double-talk": COLOR_DT}
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        for label, a, b in segments:
            if label in shades:
                ax.axvspan(a / FS, b / FS, color=shades[label], zorder=0, lw=0)
    shown = [label for label in shades if any(seg[0] == label for seg in segments)]
    if shown:
        axes[0].legend(handles=[Patch(color=shades[label], label=label) for label in shown],
                       loc="lower right", bbox_to_anchor=(1, 1), ncol=len(shown), frameon=False, fontsize=9)

    kind = "call demo" if args.call else "demo"
    fig.suptitle(f"AEC {kind} — M = {M}, L_h = {args.lh}, RT60 = {args.rt60 * 1e3:.0f} ms, ERL = {args.erl:.0f} dB, "
                 f"ENR = {args.enr:.0f} dB, no double-talk detector", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "demo.png", dpi=150)
    print(f"Saved {out / 'demo.png'}")


if __name__ == "__main__":
    main()
