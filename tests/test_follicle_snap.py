"""Reconstructing the base of a paw-occluded whisker.

The properties that matter, in order of how badly they would bite:

  * follicle_x/y is never modified -- the measured value is data, the snapped one
    is inference, and analysis has to be able to tell them apart;
  * a whisker that is not occluded is not moved;
  * the extrapolation follows the whisker's CURVE, not its tangent (a tangent
    lands the base tens of px away, which was the whole reason the obvious
    "project along the whisker angle" version was rejected);
  * a whisker that is mostly hidden is flagged but left alone rather than invented.
"""
import numpy as np
import pandas as pd
import pytest

from wwutils.classifiers.follicle_snap import add_follicle_snap, extrapolate_base


def arc(t0, t1, n=60, r=200.0, cx=0.0, cy=0.0):
    """Points along a circular arc -- a curved whisker with known geometry."""
    t = np.linspace(t0, t1, n)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def test_extrapolates_along_the_curve_not_the_tangent():
    x, y = arc(0.0, 0.6)
    # index 0 is the base end; hide the first 20 points
    seg = np.hypot(np.diff(x), np.diff(y))
    hidden = float(seg[:20].sum())

    got = extrapolate_base(x[20:], y[20:], hidden)

    assert got is not None
    err = np.hypot(got[0] - x[0], got[1] - y[0])
    assert err < 5.0, f"curve extrapolation off by {err:.1f} px"

    # the tangent is the thing this must beat, on the very same trace
    v = np.array([x[20] - x[25], y[20] - y[25]], float)
    v /= np.hypot(*v)
    tan = np.array([x[20], y[20]]) + v * hidden
    assert err < np.hypot(tan[0] - x[0], tan[1] - y[0])


def test_zero_or_negative_missing_returns_nothing():
    x, y = arc(0.0, 0.6)
    assert extrapolate_base(x, y, 0.0) is None
    assert extrapolate_base(x, y, -10.0) is None


def base_frame(n=600, displaced=(), short_by=0.0):
    """An identity with a stable base; `displaced` frames have theirs pushed away."""
    rows = []
    for f in range(n):
        x, y = arc(0.0, 0.6)
        fx, fy, length = x[0], y[0], float(np.hypot(np.diff(x), np.diff(y)).sum())
        if f in displaced:
            k = 20
            x, y = x[k:], y[k:]
            fx, fy = x[0], y[0]
            length -= short_by
        rows.append(dict(fid=f, wid=0, face_side="left", follicle_x=float(fx),
                         follicle_y=float(fy), length=length,
                         pixels_x=x.copy(), pixels_y=y.copy()))
    return pd.DataFrame(rows)


def test_measured_follicle_is_never_modified():
    df = base_frame(displaced=set(range(100, 140)), short_by=25.0)
    before = df[["follicle_x", "follicle_y"]].copy()

    out = add_follicle_snap(df)

    pd.testing.assert_frame_equal(out[["follicle_x", "follicle_y"]], before)
    assert {"base_occluded", "follicle_snap_x", "follicle_snap_y"} <= set(out.columns)


def test_clean_whisker_is_left_alone():
    out = add_follicle_snap(base_frame())

    assert not out.base_occluded.any()
    assert np.allclose(out.follicle_snap_x, out.follicle_x)
    assert np.allclose(out.follicle_snap_y, out.follicle_y)


def test_displaced_base_is_flagged_and_moved_back():
    disp = set(range(100, 140))
    df = base_frame(displaced=disp, short_by=25.0)

    out = add_follicle_snap(df)

    flagged = out[out.base_occluded]
    assert len(flagged) > 0, "displaced bases not detected"
    assert set(flagged.fid) <= disp, "flagged a frame whose base was fine"
    # the reconstructed base must be closer to the true one than the measured base
    true_x, true_y = arc(0.0, 0.6)
    tx, ty = true_x[0], true_y[0]
    for r in flagged.itertuples():
        d_meas = np.hypot(r.follicle_x - tx, r.follicle_y - ty)
        d_snap = np.hypot(r.follicle_snap_x - tx, r.follicle_snap_y - ty)
        assert d_snap < d_meas


def test_snap_columns_always_populated():
    """Downstream should be able to use the snap columns unconditionally."""
    out = add_follicle_snap(base_frame(displaced={200, 201}, short_by=25.0))
    assert out.follicle_snap_x.notna().all()
    assert out.follicle_snap_y.notna().all()


@pytest.mark.parametrize("frac", [0.5, 0.8])
def test_mostly_hidden_whisker_is_flagged_but_not_invented(frac):
    disp = set(range(100, 140))
    x, y = arc(0.0, 0.6)
    full = float(np.hypot(np.diff(x), np.diff(y)).sum())
    df = base_frame(displaced=disp, short_by=full * frac)

    out = add_follicle_snap(df)

    bad = out[out.fid.isin(disp)]
    assert bad.base_occluded.all()
    # too much missing -> left exactly as measured
    assert np.allclose(bad.follicle_snap_x, bad.follicle_x)
    assert np.allclose(bad.follicle_snap_y, bad.follicle_y)
