"""bridge_gaps must recover partially occluded whiskers without displacing anything.

A whisker crossed by the cue tip is still traced, just short, and the
`min_length_frac` floor rejected it -- the gate excluded the case the function
exists for. The rescue relaxes the length floor ONLY where the normal rule found
nothing, and only for a candidate whose base sits almost exactly where
interpolation predicts.

The "strictly a fallback" property is the important one to hold onto: a blanket
lower floor made adoption pick a short wrong candidate on base distance and block
the correct longer one, losing real whiskers on a fast-whisking clip.
"""
import numpy as np
import pandas as pd
import pytest

from wwutils.classifiers.hmm_link import bridge_gaps

GATE = 25.0


def det(fid, wid, fx, fy, length, side="left"):
    return dict(fid=fid, wid=wid, face_side=side, follicle_x=float(fx),
                follicle_y=float(fy), length=float(length), angle=0.0,
                tip_x=0.0, tip_y=0.0, score=500.0, label=int(wid))


def frames(gap_at=1):
    """One identity present at fid 0 and 2, missing at `gap_at`, base moving 0->20."""
    out = pd.DataFrame([det(0, 0, 100, 100, 100.0), det(2, 0, 120, 100, 100.0)])
    return out


def test_short_segment_at_the_right_base_is_recovered():
    out = frames()
    # a 30%-length segment sitting exactly at the interpolated base (110, 100)
    comb = pd.concat([out, pd.DataFrame([det(1, 7, 110, 100, 30.0)])],
                     ignore_index=True)
    out = comb.iloc[[0, 1]].copy()

    res = bridge_gaps(out, comb, gate_px=GATE)

    assert set(res.fid) == {0, 1, 2}, "partially occluded whisker not recovered"
    assert int(res[res.fid == 1].wid.iloc[0]) == 0


def test_short_segment_off_base_is_not_recovered():
    """The tight base gate is the whole discriminator -- it must actually bite."""
    out = frames()
    # same length, but its base is 15 px off: beyond 0.32 * 25 = 8 px
    comb = pd.concat([out, pd.DataFrame([det(1, 7, 125, 100, 30.0)])],
                     ignore_index=True)
    out = comb.iloc[[0, 1]].copy()

    res = bridge_gaps(out, comb, gate_px=GATE)

    assert set(res.fid) == {0, 2}, "adopted a short segment with a displaced base"


def test_rescue_disabled_restores_old_behaviour():
    out = frames()
    comb = pd.concat([out, pd.DataFrame([det(1, 7, 110, 100, 30.0)])],
                     ignore_index=True)
    out = comb.iloc[[0, 1]].copy()

    res = bridge_gaps(out, comb, gate_px=GATE, rescue_length_frac=0)

    assert set(res.fid) == {0, 2}


def test_rescue_never_displaces_a_normal_adoption():
    """Pass 1 owns the frame: a nearer short segment must not steal it.

    This is the regression that a blanket lower floor caused -- greedy nearest-base
    adoption took the short candidate and the full-length whisker was lost.
    """
    out = frames()
    full = det(1, 8, 114, 100, 100.0)      # full length, 4 px from interpolation
    short = det(1, 9, 110, 100, 30.0)      # shorter, and NEARER the prediction
    comb = pd.concat([out, pd.DataFrame([full, short])], ignore_index=True)
    out = comb.iloc[[0, 1]].copy()

    res = bridge_gaps(out, comb, gate_px=GATE)

    got = res[res.fid == 1]
    assert len(got) == 1
    assert float(got.length.iloc[0]) == 100.0, "short candidate displaced the real one"


@pytest.mark.parametrize("length_frac", [0.05, 0.11, 0.3])
def test_recovers_across_the_short_range(length_frac):
    out = frames()
    comb = pd.concat([out, pd.DataFrame([det(1, 7, 110, 100, 100.0 * length_frac)])],
                     ignore_index=True)
    out = comb.iloc[[0, 1]].copy()

    res = bridge_gaps(out, comb, gate_px=GATE)

    # default rescue_length_frac is 0.10, so 0.05 must NOT come back
    expected = {0, 1, 2} if length_frac >= 0.10 else {0, 2}
    assert set(res.fid) == expected


def test_gap_longer_than_max_gap_is_left_alone():
    out = pd.DataFrame([det(0, 0, 100, 100, 100.0), det(50, 0, 120, 100, 100.0)])
    rows = [det(f, 7, 100 + 0.4 * f, 100, 30.0) for f in range(1, 50)]
    comb = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)
    out = comb.iloc[[0, 1]].copy()

    res = bridge_gaps(out, comb, gate_px=GATE, max_gap=20)

    assert set(res.fid) == {0, 50}, "invented a whisker across a long occlusion"
    assert not np.isnan(res.length).any()
