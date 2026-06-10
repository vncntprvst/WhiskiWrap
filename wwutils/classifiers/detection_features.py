"""Leak-free per-detection features for the learned whisker add-on models.

Single source of truth for features so training and serving stay identical. Two model
families consume these:
  * COVERAGE (real-whisker vs noise) -- uses shape/size features that are universal
    across sessions (length, score, point count, shape descriptor, smoothness).
  * IDENTITY (which whisker, per session) -- additionally uses the stable base
    position/emergence cues (normalized follicle, base angle, length).

Design rules enforced here (see handoff "three past failures"):
  * NO `tip_x`/`tip_y` -- tips sweep wildly during whisking and overlap across whiskers.
  * NO raw `curvature` -- it is ~0 for every detection (useless). We instead derive a
    dimensionless shape ratio from the skeleton.
  * NO feature is ever computed with `groupby('wid')` over the clip (that leaks the
    label). Any per-track temporal feature must be built elsewhere from an explicit,
    causal, track-local window -- never from the predicted identity column.

The skeleton descriptor is translation- and rotation-normalized, so it encodes only the
intrinsic curve shape (shared across whiskers and sessions); identity-bearing absolutes
(follicle position, length, base angle) are kept as separate explicit columns so each
model decides what it is allowed to use.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# Feature column groups (the models select from these).
SHAPE_SCALARS: List[str] = ["arc_len", "chord_len", "curviness", "straightness", "n_pts"]
COVERAGE_FEATURES: List[str] = ["length", "score", "pixel_length"] + SHAPE_SCALARS
IDENTITY_FEATURES: List[str] = [
    "length", "follicle_x_n", "follicle_y_n", "base_angle", "angle",
    "arc_len", "chord_len", "curviness", "straightness",
]
# EXCLUDED on purpose: tip_x, tip_y, curvature.


def _as_xy(px, py) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(px, dtype=float), np.asarray(py, dtype=float)


def _orient_follicle_first(px: np.ndarray, py: np.ndarray,
                           fx: float, fy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the skeleton ordered from the follicle end to the tip end.

    The tracer's point order is not consistent (only ~56% start at the follicle), so we
    pick the endpoint nearest the measured follicle as the base.
    """
    if len(px) < 2:
        return px, py
    d0 = (px[0] - fx) ** 2 + (py[0] - fy) ** 2
    dn = (px[-1] - fx) ** 2 + (py[-1] - fy) ** 2
    if dn < d0:
        return px[::-1], py[::-1]
    return px, py


def _resample_arclen(px: np.ndarray, py: np.ndarray, k: int) -> np.ndarray:
    """Arc-length resample a polyline to ``k`` points -> (k, 2) array."""
    seg = np.hypot(np.diff(px), np.diff(py))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.column_stack([np.full(k, px[0]), np.full(k, py[0])])
    t = np.linspace(0.0, total, k)
    return np.column_stack([np.interp(t, s, px), np.interp(t, s, py)])


def skeleton_descriptor(px, py, fx: float, fy: float, k: int = 16):
    """Pose-invariant shape vector + leak-free shape scalars for one detection.

    Returns (shape_vec[2k], scalars dict). shape_vec is translated to the follicle and
    rotated so the follicle->tip chord lies on +x => invariant to where the whisker sits
    and to its whisking-phase orientation.
    """
    px, py = _as_xy(px, py)
    n = len(px)
    if n < 2:
        return np.zeros(2 * k), dict(arc_len=0.0, chord_len=0.0, curviness=1.0,
                                     straightness=0.0, base_angle=0.0, n_pts=float(n))
    px, py = _orient_follicle_first(px, py, fx, fy)
    # scalars in the ORIGINAL frame (absolute base angle is identity-relevant)
    arc_len = float(np.hypot(np.diff(px), np.diff(py)).sum())
    chord = np.array([px[-1] - px[0], py[-1] - py[0]], float)
    chord_len = float(np.hypot(*chord))
    curviness = arc_len / chord_len if chord_len > 1e-6 else 1.0
    # base emergence angle: vector from follicle to a point ~15px along the arc
    seg = np.hypot(np.diff(px), np.diff(py))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    j = int(np.searchsorted(s, min(15.0, s[-1] * 0.5)))
    j = max(1, min(j, n - 1))
    base_angle = float(np.degrees(np.arctan2(py[j] - py[0], px[j] - px[0])))
    # canonical (translate to follicle, rotate chord onto +x)
    pts = _resample_arclen(px, py, k) - np.array([px[0], py[0]])
    theta = np.arctan2(chord[1], chord[0]) if chord_len > 1e-6 else 0.0
    ct, st = np.cos(-theta), np.sin(-theta)
    rot = pts @ np.array([[ct, -st], [st, ct]]).T
    # straightness: RMS off-chord deviation (the y residual in canonical frame), scaled
    straightness = float(np.sqrt(np.mean(rot[:, 1] ** 2)) / (chord_len + 1e-6))
    scalars = dict(arc_len=arc_len, chord_len=chord_len, curviness=curviness,
                   straightness=straightness, base_angle=base_angle, n_pts=float(n))
    return rot.reshape(-1), scalars


def add_shape_features(df: pd.DataFrame, k: int = 16,
                       include_vec: bool = False) -> pd.DataFrame:
    """Add shape scalar columns (and optionally the 2k shape vector) to ``df``."""
    out = df.copy()
    rows = [skeleton_descriptor(r.pixels_x, r.pixels_y, r.follicle_x, r.follicle_y, k)
            for r in df.itertuples(index=False)]
    scal = pd.DataFrame([s for _, s in rows], index=df.index)
    for c in scal.columns:
        out[c] = scal[c].values
    if include_vec:
        vec = np.vstack([v for v, _ in rows])
        for i in range(vec.shape[1]):
            out[f"shape_{i}"] = vec[:, i]
    return out


def add_normalized_follicle(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(face_side) normalized follicle: (follicle - side median) / side median length.

    Head is fixed; this expresses 'where is the base within this side's cluster' without
    hard-coded pixels and without naming which specific whisker it is. Computed per the
    *given* dataframe (per target clip at inference), never via groupby('wid').
    """
    out = df.copy()
    out["follicle_x_n"] = np.nan
    out["follicle_y_n"] = np.nan
    for side, g in df.groupby("face_side"):
        scale = max(float(g["length"].median()), 1.0)
        out.loc[g.index, "follicle_x_n"] = (g["follicle_x"] - g["follicle_x"].median()) / scale
        out.loc[g.index, "follicle_y_n"] = (g["follicle_y"] - g["follicle_y"].median()) / scale
    return out


def detection_features(df: pd.DataFrame, k: int = 16) -> pd.DataFrame:
    """Full engineered feature frame (index-aligned to df). Adds shape scalars and the
    normalized follicle; leaves raw columns in place. Models select COVERAGE_FEATURES or
    IDENTITY_FEATURES from the result."""
    out = add_shape_features(df, k=k, include_vec=False)
    out = add_normalized_follicle(out)
    return out


if __name__ == "__main__":  # quick invariance self-test
    rng = np.random.default_rng(0)
    px = np.linspace(0, 100, 40) + rng.normal(0, 0.5, 40)
    py = 0.2 * px + rng.normal(0, 0.5, 40)
    v1, s1 = skeleton_descriptor(px, py, px[0], py[0])
    # rotate + translate -> descriptor must be ~unchanged
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p2 = (np.column_stack([px, py]) @ R.T) + np.array([37.0, -12.0])
    v2, s2 = skeleton_descriptor(p2[:, 0], p2[:, 1], p2[0, 0], p2[0, 1])
    print("shape vec max abs diff under rot+trans:", np.abs(v1 - v2).max())
    print("curviness s1=%.4f s2=%.4f (should match)" % (s1["curviness"], s2["curviness"]))
