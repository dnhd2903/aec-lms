import numpy as np
import pytest

from sim.conversation import Turn, place_timeline, split_utterances

FS = 8000


def burst(sec, rng):
    return 0.3 * rng.standard_normal(int(sec * FS))


def test_split_finds_lines_bridges_short_pauses_and_drops_clicks():
    rng = np.random.default_rng(0)
    gap = lambda sec: 1e-3 * rng.standard_normal(int(sec * FS))
    click = 0.5 * rng.standard_normal(int(0.05 * FS))
    # click | line 1 (1.0 s) | line 2: 0.6 s + 0.3 s pause + 0.6 s | line 3 (2.0 s)
    x = np.concatenate([gap(1.5), click, gap(1.5), burst(1.0, rng), gap(2.0),
                        burst(0.6, rng), gap(0.3), burst(0.6, rng), gap(2.0), burst(2.0, rng), gap(1.0)])
    utts = split_utterances(x, FS)
    assert len(utts) == 3
    for u, expected in zip(utts, [1.0, 1.5, 2.0]):
        assert len(u) / FS == pytest.approx(expected + 0.2, abs=0.05)   # + 2 × 0.1 s padding


def test_timeline_places_overlaps():
    script = [Turn("A1", None, "start", 0.5), Turn("B1", "A1", "end", -1.0), Turn("A2", "B1", "end", 0.5)]
    far, near = [np.ones(3 * FS), np.ones(FS)], [np.ones(2 * FS)]
    call = place_timeline(script, far, near, FS, tail=1.0)
    assert call.spans == {"A1": (4000, 28000), "B1": (20000, 36000), "A2": (40000, 48000)}
    assert len(call.x) == 48000 + FS
    assert call.x[20000:28000].all() and call.s[20000:28000].all()     # double-talk
    assert not call.s[:20000].any() and not call.x[36000:40000].any()


def test_timeline_rejects_wrong_line_count_and_self_overlap():
    script = [Turn("A1", None, "start", 0.0), Turn("B1", "A1", "start", 0.0), Turn("B2", "B1", "end", -0.5)]
    with pytest.raises(ValueError, match="far-end lines"):
        place_timeline(script, [], [np.ones(FS), np.ones(FS)], FS)
    with pytest.raises(ValueError, match="overlap"):
        place_timeline(script, [np.ones(FS)], [np.ones(FS), np.ones(FS)], FS)
