"""
sim/conversation.py — Assemble a two-party call from one recording per talker.

Each talker records all of their lines in one file, in script order, separated by
about 2 s of silence. split_utterances() cuts a recording into lines at the
pauses; place_timeline() puts the lines on a common timeline following a
turn-taking script (who speaks after whom, and where they overlap). The result is
the far-end track x (loudspeaker) and the near-end track s (local talker).

Call scenario (lines in data/call/script.vi.md): An (far end) calls Binh (near end),
who is on a laptop without headphones.

    A1  An opens the call          far-end single-talk, long → the filter converges
    B1  Binh answers               near-end single-talk, x ≈ 0 → nothing to cancel
    A2  An explains                far-end single-talk ...
    B2  Binh: "uh-huh"             ... short double-talk (backchannel) in the middle of A2
    B3  Binh interrupts            double-talk over the last seconds of A2
    A3  An answers
    B4  Binh proposes a meeting
    A4  An says goodbye            ... while
    B5  Binh says goodbye          double-talk
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np


class Turn(NamedTuple):
    id: str               # "A…" = far-end talker (An), "B…" = near-end talker (Binh)
    ref: Optional[str]    # turn this one is placed relative to; None = start of the call
    anchor: str           # "start", "mid" or "end" of the ref turn
    offset: float         # seconds after the anchor; negative with "end" ⇒ overlaps the ref turn


CALL_SCRIPT = [
    Turn("A1", None, "start", 0.5),
    Turn("B1", "A1", "end", 0.6),
    Turn("A2", "B1", "end", 0.5),
    Turn("B2", "A2", "mid", -0.5),
    Turn("B3", "A2", "end", -2.0),
    Turn("A3", "B3", "end", 0.4),
    Turn("B4", "A3", "end", 0.5),
    Turn("A4", "B4", "end", 0.5),
    Turn("B5", "A4", "end", -1.0),
]


class Call(NamedTuple):
    x: np.ndarray                          # far-end track
    s: np.ndarray                          # near-end track
    spans: Dict[str, Tuple[int, int]]      # turn id → (start, end) sample


def split_utterances(
    x: np.ndarray,
    fs: int,
    min_silence: float = 1.0,
    min_length: float = 0.25,
    pad: float = 0.1,
    frame: float = 0.02,
) -> List[np.ndarray]:
    """
    Cut a recording into utterances at pauses of at least min_silence seconds.

    Frame energies are compared with a threshold placed between the noise floor
    (10th percentile) and the speech level (99th percentile). Pauses shorter than
    min_silence (between words) are bridged; active runs shorter than min_length
    (clicks, breaths) are dropped. Each utterance keeps pad seconds of context on
    both sides so soft onsets and tails are not cut.
    """
    hop = int(frame * fs)
    n_frames = len(x) // hop
    if n_frames == 0:
        return []
    e_db = 10 * np.log10(np.mean(x[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1) + 1e-12)
    noise_db, speech_db = np.percentile(e_db, [10, 99])
    active = e_db > noise_db + max(6.0, 0.25 * (speech_db - noise_db))

    # Run boundaries: starts[i] … ends[i] (exclusive) are the active runs, in frames.
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]

    runs = []
    for a, b in zip(starts, ends):
        if runs and (a - runs[-1][1]) * frame < min_silence:
            runs[-1][1] = b                       # bridge a short pause
        else:
            runs.append([a, b])
    runs = [(a, b) for a, b in runs if (b - a) * frame >= min_length]

    p = int(pad * fs)
    return [x[max(0, a * hop - p): min(len(x), b * hop + p)].copy() for a, b in runs]


def place_timeline(
    script: List[Turn],
    far_utts: List[np.ndarray],
    near_utts: List[np.ndarray],
    fs: int,
    tail: float = 1.0,
) -> Call:
    """
    Place the utterances on one timeline. The k-th "A" turn of the script gets the
    k-th far-end utterance, the k-th "B" turn the k-th near-end utterance.
    """
    a_turns = [t for t in script if t.id.startswith("A")]
    b_turns = [t for t in script if t.id.startswith("B")]
    if len(far_utts) != len(a_turns):
        raise ValueError(f"the script has {len(a_turns)} far-end lines but {len(far_utts)} were found in the recording")
    if len(near_utts) != len(b_turns):
        raise ValueError(f"the script has {len(b_turns)} near-end lines but {len(near_utts)} were found in the recording")
    audio = dict(zip([t.id for t in a_turns], far_utts)) | dict(zip([t.id for t in b_turns], near_utts))

    spans: Dict[str, Tuple[int, int]] = {}
    for t in script:
        if t.ref is None:
            base = 0
        elif t.ref not in spans:
            raise ValueError(f"turn {t.id} refers to {t.ref}, which must come earlier in the script")
        else:
            a, b = spans[t.ref]
            base = {"start": a, "mid": (a + b) // 2, "end": b}[t.anchor]
        start = base + int(t.offset * fs)
        if start < 0:
            raise ValueError(f"turn {t.id} would start before the call begins")
        spans[t.id] = (start, start + len(audio[t.id]))

    # One talker cannot say two lines at once.
    for prefix in ("A", "B"):
        own = sorted((span, tid) for tid, span in spans.items() if tid.startswith(prefix))
        for (s1, id1), (s2, id2) in zip(own, own[1:]):
            if s2[0] < s1[1]:
                raise ValueError(f"turns {id1} and {id2} overlap ({(s1[1] - s2[0]) / fs:.1f} s): "
                                 f"line {id1} is longer than the script allows")

    N = max(b for _, b in spans.values()) + int(tail * fs)
    x, s = np.zeros(N), np.zeros(N)
    for tid, (a, b) in spans.items():
        (x if tid.startswith("A") else s)[a:b] += audio[tid]
    return Call(x=x, s=s, spans=spans)
