"""Each traced segment must carry ITS OWN measurement, never a neighbour's.

The regression these guard against: measurements were paired to traces by walking
both in parallel, with the ordering assumption checked on frame 0 only. A chunk
whose first frame happened to agree kept a silent off-by-one on every later frame,
so rows came out with a follicle, angle and length belonging to a different
whisker than their trace. It was invisible in aggregate -- the row counts and the
schema were right -- and only showed up as traces drawn on the wrong whisker in
the labelling GUI. Measured on a WA014 clip: 130 of 640 rows wrong in one chunk.

The second regression: unmatched segments were mapped to index -1, and -1 is a
valid index, so they silently took the LAST row of the measurements file.
"""
import numpy as np
import pytest

from WhiskiWrap.base import index_measurements


class FakeSeg:
    def __init__(self, time, wid, n=5):
        self.time = time
        self.id = wid
        self.x = np.arange(n, dtype=float) + 100.0 * wid
        self.y = np.arange(n, dtype=float) + 100.0 * wid


def whiskers_dict(frames):
    """{frame: {wid: seg}} from {frame: [wid, ...]}."""
    return {f: {w: FakeSeg(f, w) for w in wids} for f, wids in frames.items()}


def measurement_row(frame, wid):
    """A measurements row: col 1 is frame, col 2 is wid, rest identifiable by wid."""
    m = np.zeros(11, dtype=float)
    m[0] = 0          # label
    m[1] = frame
    m[2] = wid
    m[3:] = wid       # every other field carries the wid, so a mispairing shows
    return m


def test_reorder_when_only_a_later_frame_disagrees():
    """Frame 0 in order, frame 1 shuffled -- the case the old frame-0 check missed."""
    whiskers = whiskers_dict({0: [0, 1, 2], 1: [0, 1, 2]})
    # measurements: frame 0 in trace order, frame 1 reversed
    measurements = np.array(
        [measurement_row(0, 0), measurement_row(0, 1), measurement_row(0, 2),
         measurement_row(1, 2), measurement_row(1, 1), measurement_row(1, 0)])

    out = index_measurements(whiskers, measurements)

    # iteration order is frame 0 wids 0,1,2 then frame 1 wids 0,1,2
    assert [int(r[2]) for r in out] == [0, 1, 2, 0, 1, 2]
    # and the payload columns must travel with the wid, not stay in file order
    for row in out:
        assert np.all(row[3:] == row[2])


def test_missing_segment_is_nan_not_the_last_row():
    """A segment absent from .measurements must not inherit another whisker's numbers."""
    whiskers = whiskers_dict({0: [0, 1, 9]})          # wid 9 has no measurement
    measurements = np.array([measurement_row(0, 0), measurement_row(0, 1)])

    out = index_measurements(whiskers, measurements)

    assert np.all(out[0][3:] == 0)
    assert np.all(out[1][3:] == 1)
    # the unmatched one is explicitly absent -- previously it took measurements[-1]
    assert np.all(np.isnan(out[2])), "unmatched segment inherited another row"


def test_already_aligned_input_is_unchanged():
    """Applying the alignment unconditionally must be a no-op when nothing is wrong."""
    whiskers = whiskers_dict({0: [0, 1], 1: [0, 1]})
    measurements = np.array(
        [measurement_row(0, 0), measurement_row(0, 1),
         measurement_row(1, 0), measurement_row(1, 1)])

    out = index_measurements(whiskers, measurements)

    assert np.array_equal(out, measurements)


@pytest.mark.parametrize("shift", [1, 2])
def test_offset_measurements_are_repaired(shift):
    """A whole-file positional shift is exactly what used to go undetected."""
    frames = {f: [0, 1, 2] for f in range(4)}
    whiskers = whiskers_dict(frames)
    rows = [measurement_row(f, w) for f in frames for w in frames[f]]
    # rotate: every segment would otherwise be paired `shift` rows off
    measurements = np.array(rows[shift:] + rows[:shift])

    out = index_measurements(whiskers, measurements)

    expected = [w for f in frames for w in frames[f]]
    assert [int(r[2]) for r in out] == expected
    for row in out:
        assert np.all(row[3:] == row[2])
