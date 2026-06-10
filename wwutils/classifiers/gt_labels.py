"""Manufacture supervised labels by matching combined detections to ground truth.

The combined parquet holds every traced detection (real whiskers + noise). Ground truth
holds the confirmed real whiskers with their identities. Matching each combined detection
to the nearest GT detection (per frame+side, Hungarian, tight gate) yields, with zero hand
labeling beyond the one GT clip:
  * ``is_real`` (bool)  -- coverage target (matched = real).
  * ``gt_wid`` (int)    -- identity target for matched reals (-1 for noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match_to_gt(combined: pd.DataFrame, gt: pd.DataFrame, *,
                gate_px: float = 12.0) -> pd.DataFrame:
    """Return ``combined`` with added ``is_real`` (bool) and ``gt_wid`` (int, -1 noise).

    Matching is per (fid, face_side) by follicle distance, one-to-one, gated. The gate is
    tighter than the evaluator's (25 px) because here we need precise real/noise labels.
    """
    out = combined.copy()
    out["is_real"] = False
    out["gt_wid"] = -1
    gt_by = {key: g for key, g in gt.groupby(["fid", "face_side"])}
    for key, c in combined.groupby(["fid", "face_side"]):
        g = gt_by.get(key)
        if g is None or g.empty:
            continue
        cx = c[["follicle_x", "follicle_y"]].to_numpy(float)
        gx = g[["follicle_x", "follicle_y"]].to_numpy(float)
        d = np.sqrt(((cx[:, None, :] - gx[None, :, :]) ** 2).sum(-1))
        ci, gi = linear_sum_assignment(d)
        for r, k in zip(ci, gi):
            if d[r, k] <= gate_px:
                idx = c.index[r]
                out.at[idx, "is_real"] = True
                out.at[idx, "gt_wid"] = int(g.iloc[k]["wid"])
    return out
