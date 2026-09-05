"""Cross-chunk stitching must preserve identity.

whisk numbers identities independently inside each traced chunk;
``stitch_chunk_identities`` joins them into one global identity per whisker. When
it fails, the count of global ids explodes (60, 129 and 678 ids were observed on
real sessions whose classify labels said 5, 8 and 12) and -- worse, because the
downstream coverage filter hides it -- a single whisker gets carried as two ids
either side of a boundary, so its angle trace breaks in the middle.

These tests build chunks whose correct answer is known and assert the global ids
come back right.

WHY THE COST NEEDS ANGLE, AND WHY AT THE SEAM
    Bases on one pad are a few px apart, so position alone cannot separate them.
    And a seam is between two ADJACENT frames: whiskers protract at ~1.5-3 deg per
    frame, so smoothing the angle over even a few frames moves it further than the
    gap between two whiskers. Both are asserted below.
"""
import numpy as np
import pandas as pd
import pytest

from wwutils.classifiers.hmm_link import stitch_chunk_identities


def _chunks(df, chunk):
    """Split by frame and renumber identities inside each chunk, as classify does."""
    out = []
    for c0 in range(0, int(df.fid.max()) + 1, chunk):
        blk = df[(df.fid >= c0) & (df.fid < c0 + chunk)].copy()
        if blk.empty:
            continue
        order = blk.groupby("gt_wid").follicle_y.median().sort_values().index.tolist()
        blk["state"] = blk.gt_wid.map({w: i for i, w in enumerate(order)})
        out.append((c0, blk))
    return out


def _accuracy(res):
    """Best one-to-one truth<->gid agreement, as a fraction of detections."""
    from scipy.optimize import linear_sum_assignment
    real = res[res.gt_wid >= 0]
    tab = pd.crosstab(real.gt_wid, real.gid)
    ri, ci = linear_sum_assignment(-tab.to_numpy())
    return tab.to_numpy()[ri, ci].sum() / len(real)


def _whisking(n_frames=400, chunk=100, n_whiskers=2, sweep_deg_per_frame=2.0,
              base_sep_px=6.0, base_noise_px=7.0, angle_sep_deg=16.0, seed=0):
    """Two whiskers on one pad, whisking together, with a NOISY base estimate.

    The noise is the point. With a clean base a positional cost separates whiskers
    6 px apart perfectly and every test here passes whether or not the cost uses
    angle -- which would make them prove nothing.

    Real bases are not clean: on the hand-corrected clip a whisker's reported
    follicle moved 20 px between two adjacent frames while the two whiskers' bases
    were only 7 px apart, because whisk's follicle is one end of a traced segment
    and moves with how much of the whisker was traced. So ``base_noise_px`` is set
    at the same scale as ``base_sep_px``, which is what makes position ambiguous
    and angle necessary.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fid in range(n_frames):
        phase = sweep_deg_per_frame * fid
        for w in range(n_whiskers):
            rows.append(dict(
                fid=fid, gt_wid=w, face_side="left", state=0,
                follicle_x=100.0 + rng.normal(0, base_noise_px),
                follicle_y=60.0 + w * base_sep_px + rng.normal(0, base_noise_px),
                angle=(-80.0 + w * angle_sep_deg + phase + rng.normal(0, 0.5)),
                length=150.0 - 10.0 * w, curvature=0.0, score=500.0,
                tip_x=0.0, tip_y=0.0, chunk_start=(fid // chunk) * chunk))
    return pd.DataFrame(rows)


def test_two_clean_whiskers_keep_one_id_each(synth_two_chunks):
    df = synth_two_chunks.rename(columns={"gt_wid": "gt_wid"})
    res = stitch_chunk_identities(_chunks(df, 5))
    assert not res.empty
    assert res.gid.nunique() == 2, "two whiskers must produce exactly two global ids"
    assert _accuracy(res) == 1.0


def test_close_bases_are_separated_by_angle():
    """Bases 6 px apart, angles 16 deg apart, both sweeping.

    Position alone cannot do this; if the cost ever drops the angle term, the two
    whiskers swap at a seam and this fails.
    """
    df = _whisking()
    res = stitch_chunk_identities(_chunks(df, 100))
    assert res.gid.nunique() == 2, f"expected 2 ids, got {res.gid.nunique()}"
    assert _accuracy(res) == 1.0


def test_identity_survives_an_absence():
    """A whisker missing from a whole chunk must rejoin its own identity.

    Matching only against the previous chunk cannot do this -- the whisker is
    unmatched, mints a fresh id, and the id count grows for the rest of the
    session. That is the mechanism behind the 678-id session.
    """
    df = _whisking(n_frames=400, chunk=100)
    df = df[~((df.gt_wid == 1) & (df.fid >= 100) & (df.fid < 200))]
    res = stitch_chunk_identities(_chunks(df, 100))
    assert res.gid.nunique() == 2, (
        f"a whisker that disappears for one chunk and returns must keep its id; "
        f"got {res.gid.nunique()} ids")
    assert _accuracy(res) == 1.0


def test_a_spurious_detection_does_not_steal_an_identity():
    """An extra detection in one chunk must not take a real whisker's id.

    Hungarian always returns a full assignment, so without a distance gate the
    spurious state is guaranteed to be matched to something.
    """
    df = _whisking(n_frames=400, chunk=100)
    stub = df[(df.gt_wid == 0) & (df.fid >= 200) & (df.fid < 300)].copy()
    stub["gt_wid"] = -1                      # not a whisker
    stub["follicle_y"] = stub["follicle_y"] + 25.0
    stub["angle"] = stub["angle"] + 60.0
    stub["length"] = 40.0
    df = pd.concat([df, stub], ignore_index=True)
    res = stitch_chunk_identities(_chunks(df, 100))
    assert _accuracy(res) == 1.0, "a fur stub took a real whisker's identity"


@pytest.mark.parametrize("smooth", [1, 2, 5])
def test_angle_window_at_the_seam(smooth):
    """Only a one-frame angle window is safe when whiskers sweep.

    Documents the measurement that set the default: with whiskers protracting
    2 deg/frame and sitting 16 deg apart, smoothing over 5 frames moves a whisker
    10 deg -- comparable to the separation -- and identity is lost.
    """
    df = _whisking(sweep_deg_per_frame=2.0, angle_sep_deg=16.0)
    res = stitch_chunk_identities(_chunks(df, 100), angle_frames=smooth)
    acc = _accuracy(res)
    if smooth == 1:
        assert acc == 1.0
    # larger windows are not asserted to fail -- the point is that 1 is safe
