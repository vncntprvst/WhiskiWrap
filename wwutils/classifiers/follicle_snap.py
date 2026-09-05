"""Recover the base of a whisker whose proximal end is hidden by the paw.

THE PROBLEM
    When the forepaw crosses a whisker, whisk traces the part it can see. The
    distal shaft is found correctly -- the angle and shape are right, which is the
    information the science uses -- but the trace's near end stops at the paw edge,
    so the reported follicle sits on the paw instead of on the face. Measured on
    WA015, one whisker had its base more than 15 px from its own rolling-median
    base position in 25% of frames, against 2-7 px for every unoccluded identity.

    That is not only a bookkeeping error. The follicle is an input, not just an
    output: the follicle-outlier filter drops detections far from their identity's
    base, gap bridging interpolates base position, and the identity re-ranker uses
    normalised follicle displacement as a feature. A base displaced onto the paw
    degrades all three.

WHAT DOES NOT WORK
    Projecting from the trace's near end along the whisker's own angle -- the
    intuitive "snap to the face contour". Whiskers curve, so a local tangent does
    not extrapolate back to the base: measured against traces with a known-good
    follicle, cut short and projected back, the tangent lands a median 22-50 px
    away, which is the same size as the displacement being corrected.

    Nor is a face contour easy to come by. The paw is as dark as the snout, so a
    single-frame silhouette segments the two as one blob; and a temporal median
    over a 300-frame clip does not remove the paw, because it rests in place for
    seconds at a time. It only washes out over thousands of frames.

WHAT WORKS
    Extrapolating the traced SHAPE rather than its tangent, and by the whisker's
    own missing length rather than to any external contour -- so no face model is
    needed at all. Fitting a low-order curve in the trace's own frame and walking
    it out by the missing ARCLENGTH (not axis distance -- the two differ by exactly
    the curvature being fitted) puts the base back accurately.

    Measured on WA012/WA015/WA016 by cutting traces with a known-good base and
    asking for the base back, degree 2:

        hidden 15%   median 0.8 px      99% within 5 px
        hidden 25%   median 1.2-2.1     98-99% within 5 px
        hidden 30%   median 1.6-2.6     94-97% within 5 px
        hidden 40%   median 2.2-3.5     86-100% within 10 px
        hidden 45%   median 3.2-4.5     70-98% within 10 px
        hidden 50%   median 3.8-7.1     degrades

    Degree 2, not 3. Cubic matches it at the median but has a much worse tail
    (p90 5.3-10.0 px vs 3.0-4.5): the extra freedom fits wiggle in the retained
    segment and that swings once extrapolated. The tangent is worse again --
    median 2.6 px but p90 7.4 and only 62% within 5 px.

    The cap is on ``missing / retained`` -- how far we extrapolate per unit of
    visible curve -- rather than on the fraction of the whisker hidden. That is the
    quantity accuracy actually tracks, and unlike a fraction-hidden cap it behaves
    the same across animals:

        ratio <= 0.50    median 1.2-2.1 px    100% within 10 px
        ratio 0.50-0.75  median 1.9-3.0       98-100% within 10 px
        ratio 0.75-1.00  median 3.7-4.4       74-97% within 10 px
        ratio > 1.00     median 3.6-10.4      47-96% within 10 px

    0.75 is where every animal is still at 98%+; past 1.0 WA012 falls to 47% while
    WA015 is still at 96%, so a fraction-hidden cap tuned on one animal would be
    wrong for another. Beyond the cap the detection is flagged and its follicle
    left exactly as measured -- the trace, and so the angle, is still good, and is
    the reason such a detection is worth keeping rather than deleting.

WHAT IS WRITTEN
    Three columns, and ``follicle_x``/``follicle_y`` are never modified:

        base_occluded    the base is displaced -- treat the follicle with suspicion
        follicle_snap_x  reconstructed base, or the measured one where no
        follicle_snap_y  correction applied

    The measured follicle is data; the snapped one is inference, and the two must
    stay distinguishable. Anything auditing detection quality needs the raw value;
    analysis that wants a base in the right place uses the snap columns, which are
    always populated so they can be used uniformly.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

WINDOW = 401           # frames; must outlast a resting paw
MIN_DEV_PX = 20.0      # floor for "displaced", from measured base stability
DEV_K = 4.0            # ... or this many times the identity's own median deviation
MAX_EXTRAP_RATIO = 0.75   # missing / retained arclength
FIT_POINTS = 40


def _oriented(px: np.ndarray, py: np.ndarray, fx: float, fy: float):
    """Return the trace ordered so index 0 is the follicle end."""
    if np.hypot(px[0] - fx, py[0] - fy) > np.hypot(px[-1] - fx, py[-1] - fy):
        return px[::-1], py[::-1]
    return px, py


def extrapolate_base(px: np.ndarray, py: np.ndarray, missing: float,
                     deg: int = 2, fit_points: int = FIT_POINTS
                     ) -> Optional[Tuple[float, float]]:
    """Extend a trace backwards past its near end by ``missing`` px of arclength.

    ``px[0]``/``py[0]`` must be the near (base-side) end. The fit is done in the
    trace's own frame -- arclength along the chord vs lateral offset -- because a
    whisker is close to single-valued there and not in image coordinates.
    """
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    if len(px) < 6 or not np.isfinite(missing) or missing <= 0:
        return None
    use = min(len(px), max(fit_points, deg + 3))
    px, py = px[:use], py[:use]

    k = min(len(px) - 1, 20)
    ax = np.array([px[0] - px[k], py[0] - py[k]], float)
    n = np.hypot(*ax)
    if n < 1e-6:
        return None
    ax /= n
    ay = np.array([-ax[1], ax[0]])
    o = np.array([px[0], py[0]])
    d = np.stack([px - o[0], py - o[1]], 1)
    s = d @ ax                      # <= 0 over the retained trace
    u = d @ ay

    # drop to a degree the data can support rather than fitting noise
    while deg > 1 and len(px) < deg + 3:
        deg -= 1
    try:
        c = np.polyfit(s, u, deg)
    except (np.linalg.LinAlgError, ValueError):
        return None

    # walk forward accumulating ARCLENGTH, not axis distance: the two differ by
    # exactly the curvature we went to the trouble of fitting
    step = 0.5
    acc = 0.0
    s_prev, u_prev = 0.0, float(np.polyval(c, 0.0))
    s_cur = 0.0
    for _ in range(int(missing / step) * 4 + 8):
        s_cur += step
        u_cur = float(np.polyval(c, s_cur))
        acc += float(np.hypot(s_cur - s_prev, u_cur - u_prev))
        s_prev, u_prev = s_cur, u_cur
        if acc >= missing:
            p = o + ax * s_cur + ay * u_cur
            return float(p[0]), float(p[1])
    return None


def reference_from(df: pd.DataFrame, min_dev_px: float = MIN_DEV_PX
                   ) -> dict:
    """Per-identity {(side, wid): (base_x, base_y, expected_length, scale)}.

    Two passes: a plain median, then a re-median over only the detections near it.
    One pass is not enough when a whisker is occluded for much of the window -- the
    displaced bases drag the median toward the paw, which is precisely the case
    this is for.

    ``scale`` is the spread of the INLIER bases -- how much this whisker's base
    wanders when nothing is on top of it. It is carried here rather than measured
    at use because measuring it on the target clip is circular: a whisker whose
    base is displaced in most of a clip's frames has a large spread there, which
    inflates any threshold derived from it until the displacement no longer
    exceeds it. That is not hypothetical -- it is how the first version of this
    silently flagged nothing on the one whisker it was written for.
    """
    ref = {}
    for (side, wid), g in df.groupby(["face_side", "wid"]):
        fx = g["follicle_x"].to_numpy(float)
        fy = g["follicle_y"].to_numpy(float)
        cx, cy = float(np.median(fx)), float(np.median(fy))
        d = np.hypot(fx - cx, fy - cy)
        keep = d <= max(min_dev_px, float(np.median(d)))
        if keep.sum() >= 5:
            cx, cy = float(np.median(fx[keep])), float(np.median(fy[keep]))
            d = np.hypot(fx - cx, fy - cy)
            keep = d <= max(min_dev_px, float(np.median(d)))
        ln = g["length"].to_numpy(float) if "length" in g else np.array([np.nan])
        exp = float(np.nanmedian(ln[keep])) if keep.any() else float(np.nanmedian(ln))
        scale = float(np.median(d[keep])) if keep.any() else float(np.median(d))
        ref[(side, int(wid))] = (cx, cy, exp, scale)
    return ref


def add_follicle_snap(df: pd.DataFrame, *, window: int = WINDOW,
                      min_dev_px: float = MIN_DEV_PX, dev_k: float = DEV_K,
                      max_extrap_ratio: float = MAX_EXTRAP_RATIO,
                      reference: Optional[dict] = None,
                      verbose: bool = False) -> pd.DataFrame:
    """Add ``base_occluded`` and ``follicle_snap_x/y``. Never alters follicle_x/y.

    ``reference`` -- from :func:`reference_from` on a LONGER recording -- supplies
    the resting base and full length per identity. Pass it whenever the input is a
    short excerpt: the rolling median assumes displaced frames are the minority,
    and on a 300-frame clip they need not be. One clip here has a whisker whose
    base is displaced in 67% of frames, where a self-referential estimate would
    take the paw position as the truth and flag nothing.
    """
    out = df.copy()
    if not {"pixels_x", "pixels_y", "follicle_x", "follicle_y",
            "face_side", "wid", "fid"} <= set(out.columns):
        raise KeyError("add_follicle_snap needs fid, wid, face_side, follicle_x/y "
                       "and pixels_x/y")

    out["base_occluded"] = False
    out["follicle_snap_x"] = out["follicle_x"].astype(float)
    out["follicle_snap_y"] = out["follicle_y"].astype(float)

    # --- per-frame rigid head offset -------------------------------------------
    # The base of a whisker moves for two quite different reasons: the whole head
    # shifts, and that one whisker is displaced onto an occluder. Only the second
    # is what this function is looking for, and a fixed per-identity base cannot
    # tell them apart -- on an sc012 poke clip the head sat 30-97 px from its
    # session-wide position for the entire clip and ALL SIX whiskers were flagged
    # in frames where nothing was covering them.
    #
    # The head motion is shared by every identity, so estimate it per frame as the
    # MEDIAN offset across identities and subtract it. A median over 4-6 whiskers
    # is unmoved by one or two of them being displaced, which is the case that
    # matters: on WA015 one whisker is off its base for 67% of a clip and the
    # other three still pin the head.
    offset = {}
    if reference:
        for f, g in out.groupby("fid"):
            dx, dy = [], []
            for r in g.itertuples():
                b = reference.get((r.face_side, int(r.wid)))
                if b is not None:
                    dx.append(float(r.follicle_x) - b[0])
                    dy.append(float(r.follicle_y) - b[1])
            if dx:
                offset[int(f)] = (float(np.median(dx)), float(np.median(dy)))

    n_flag = n_snap = 0
    for (side, wid), g in out.groupby(["face_side", "wid"]):
        g = g.sort_values("fid")
        fx = g["follicle_x"].to_numpy(float)
        fy = g["follicle_y"].to_numpy(float)
        # Rolling median base. A plain per-identity median would be dragged by a
        # long occlusion episode; a window this wide outlasts one but still
        # follows genuine slow drift of the animal.
        lengths = g["length"].to_numpy(float) if "length" in g else None
        if lengths is None:
            continue

        ref = (reference or {}).get((side, int(wid)))
        mx = my = None
        if ref is not None:
            exp_len = float(ref[2])
            # scale from the reference recording, NOT from `dev` on this clip
            thresh = max(min_dev_px, dev_k * float(ref[3]))
            # the identity's base, moved to where the head is in each frame
            fids = g["fid"].to_numpy()
            ox = np.array([offset.get(int(t), (0.0, 0.0))[0] for t in fids])
            oy = np.array([offset.get(int(t), (0.0, 0.0))[1] for t in fids])
            mx = ref[0] + ox
            my = ref[1] + oy
            dev = np.hypot(fx - mx, fy - my)
        else:
            mx = pd.Series(fx).rolling(window, center=True,
                                       min_periods=25).median().to_numpy()
            my = pd.Series(fy).rolling(window, center=True,
                                       min_periods=25).median().to_numpy()
            dev = np.hypot(fx - mx, fy - my)
            med_dev = float(np.nanmedian(dev)) if np.isfinite(dev).any() else 0.0
            thresh = max(min_dev_px, dev_k * med_dev)
            # expected full length from the frames whose base is NOT displaced, so
            # an occluded stretch cannot lower the bar it is measured against
            clean = dev <= thresh
            if not clean.any():
                continue
            exp_len = float(np.nanmedian(lengths[clean]))
        if not np.isfinite(exp_len) or exp_len <= 0:
            continue

        for pos, idx in enumerate(g.index):
            if not (dev[pos] > thresh):
                continue
            out.at[idx, "base_occluded"] = True
            n_flag += 1
            retained = float(lengths[pos])
            missing = exp_len - retained
            if missing <= 0 or missing > max_extrap_ratio * max(retained, 1e-6):
                continue                     # nothing to add, or too much to invent
            # Orient toward the identity's RESTING BASE, not toward this row's own
            # recorded follicle. Whisk's follicle is simply one end of the traced
            # segment, and on an occluded whisker it can be the distal one -- in
            # which case extrapolating "outward from the follicle" runs away from
            # the face and puts the reconstructed base past the tip. Measured on an
            # sc012 poke clip before this fix: 196 of 304 reconstructions moved the
            # base FARTHER from where that whisker's base actually sits.
            anchor = (mx[pos], my[pos]) if mx is not None else (ref[0], ref[1])
            px, py = _oriented(np.asarray(g.at[idx, "pixels_x"], float),
                               np.asarray(g.at[idx, "pixels_y"], float),
                               anchor[0], anchor[1])
            p = extrapolate_base(px, py, missing)
            if p is None:
                continue
            # A reconstruction that lands FARTHER from the resting base than the
            # measured follicle has gone the wrong way. Keep the measurement.
            if (np.hypot(p[0] - anchor[0], p[1] - anchor[1])
                    > np.hypot(fx[pos] - anchor[0], fy[pos] - anchor[1])):
                continue
            out.at[idx, "follicle_snap_x"] = p[0]
            out.at[idx, "follicle_snap_y"] = p[1]
            n_snap += 1

    if verbose:
        print(f"[follicle_snap] {n_flag} detections with a displaced base, "
              f"{n_snap} reconstructed ({n_flag - n_snap} left as measured: "
              f"too much of the whisker hidden, or nothing missing).")
    return out
