"""COVERAGE model: a learned real-whisker-vs-noise classifier (the missing-whisker fix).

The hand-tuned filters (length/follicle/angle) drop real, visible whiskers. This replaces
them with a learned multivariate decision and additionally re-admits real detections that
whisk left unassigned. Real-vs-noise is a universal physical distinction (a long smooth arc
vs a short jagged fragment), and all features are scale-free or per-clip normalized, so the
model is trained once on the GT clip and applied to any clip.

Model: sklearn HistGradientBoostingClassifier on engineered scalar features (clean bimodal
signal in length/score/point-count + shape scalars). No GPU needed; trains in seconds.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import detection_features as dF
from .gt_labels import match_to_gt

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False


def _features(df: pd.DataFrame, k: int = 16, features=None) -> pd.DataFrame:
    """Feature matrix for the named columns (default: the hand-crafted set).

    A missing column raises rather than being filled or dropped: a bundle trained
    with the whiskerness features applied to a parquet that lacks them would
    otherwise score every detection off a silently different feature space.
    """
    features = list(features or dF.COVERAGE_FEATURES)
    feats = dF.detection_features(df, k=k)
    for c in features:
        if c not in feats.columns and c in df.columns:
            feats[c] = df[c].to_numpy()
    missing = [c for c in features if c not in feats.columns]
    if missing:
        raise KeyError(
            f"coverage model needs {missing}, which are not on this parquet -- "
            f"run whiskerness_score.py first, or use a bundle trained without them")
    return feats[features]


def train_coverage(combined: pd.DataFrame, gt: pd.DataFrame, *,
                   fids: Optional[np.ndarray] = None, k: int = 16,
                   gate_px: float = 12.0, extra_features=None) -> Dict:
    """Train the real/noise classifier. Labels come from matching combined->GT.

    ``fids`` restricts TRAINING frames (held-out frames are excluded to avoid leakage).
    Returns a bundle dict with the fitted model and metadata.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required for the coverage model")
    train = combined if fids is None else combined[combined["fid"].isin(set(fids))]
    labeled = match_to_gt(train, gt, gate_px=gate_px)
    feat_names = list(dF.COVERAGE_FEATURES) + list(extra_features or [])
    X = _features(labeled, k=k, features=feat_names)
    y = labeled["is_real"].to_numpy(int)
    # balance the ~8% positive class
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1,
                                         max_iter=300, l2_regularization=1.0,
                                         early_stopping=True, random_state=0)
    clf.fit(X, y, sample_weight=w)
    return {"model": clf, "features": feat_names, "k": k,
            "n_train": len(y), "pos_rate": float(y.mean())}


def train_coverage_multi(pairs, *, k: int = 16, gate_px: float = 12.0,
                         extra_features=None) -> Dict:
    """Train one production coverage model from several clips.

    ``pairs`` = list of (combined_df, gt_df). Pooling multiple animals makes the universal
    real-vs-noise boundary more robust. Labels come from matching each clip's combined->GT.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required for the coverage model")
    feat_names = list(dF.COVERAGE_FEATURES) + list(extra_features or [])
    Xs, ys = [], []
    for comb, gt in pairs:
        labeled = match_to_gt(comb, gt, gate_px=gate_px)
        Xs.append(_features(labeled, k=k, features=feat_names))
        ys.append(labeled["is_real"].to_numpy(int))
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=300,
                                         l2_regularization=1.0, early_stopping=True,
                                         random_state=0)
    clf.fit(X, y, sample_weight=w)
    return {"model": clf, "features": feat_names, "k": k,
            "n_train": len(y), "pos_rate": float(y.mean()), "n_clips": len(pairs)}


def predict_real(df: pd.DataFrame, bundle: Dict) -> np.ndarray:
    """Return p(real) for each row of ``df``."""
    X = _features(df, k=bundle.get("k", 16),
                  features=bundle.get("features"))
    return bundle["model"].predict_proba(X)[:, 1]


def save_coverage(bundle: Dict, path: str) -> None:
    joblib.dump(bundle, path)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2)


def load_coverage(path: str) -> Dict:
    return joblib.load(path)


def apply_coverage_model(out: pd.DataFrame, combined: pd.DataFrame, bundle: Dict, *,
                         side_faces, gate_px: float = 25.0,
                         keep_threshold: float = 0.5,
                         admit_threshold: float = 0.9,
                         admit: bool = True) -> pd.DataFrame:
    """Coverage selection + recovery (replaces the 3 outlier filters).

    (1) Keep rows of ``out`` with p(real) >= keep_threshold (drop learned noise).
    (2) If ``admit``: for each identity's gap frames, admit a high-p(real) combined
        detection (not already assigned) near the identity's interpolated follicle, giving
        it that identity. Never invents a new wid.
    """
    if out.empty:
        return out
    out = out.copy()
    out["_p"] = predict_real(out, bundle)
    kept = out[out["_p"] >= keep_threshold].drop(columns="_p")
    if not admit:
        return kept

    used = set(zip(kept["fid"].astype(int), kept["follicle_x"].round(2),
                   kept["follicle_y"].round(2)))

    # WORK OUT WHICH FRAMES CAN ADMIT ANYTHING BEFORE SCORING ANYTHING.
    #
    # p(real) is not cheap: predict_real builds a skeleton descriptor per detection
    # (a resample and a rotation, one Python call per row), so scoring the whole
    # combined table means ~23 million of them on a real session -- tens of minutes,
    # and it was the single largest remaining cost in linking.
    #
    # Almost none of it is used. A combined row can only be admitted if it sits in a
    # frame where some identity has a GAP, and gaps are a few percent of frames. The
    # gap set is knowable from `kept` alone, so it is computed first and the model
    # runs only on rows that can actually be consulted. Rows never looked at cannot
    # change the result, so this is a pure saving rather than an approximation.
    plans = []          # (side, wid, missing frames, expected x, expected y, med_len)
    needed = set()
    for side in kept["face_side"].unique():
        for wid, g in kept[kept["face_side"] == side].groupby("wid"):
            g = g.sort_values("fid")
            fol = g.drop_duplicates("fid").set_index("fid")[["follicle_x", "follicle_y"]]
            fids = fol.index.to_numpy()
            if fids.size < 2:
                continue
            span = np.arange(int(fids[0]), int(fids[-1]) + 1)
            missing = span[~np.isin(span, fids)]
            if missing.size == 0:
                continue
            fx = fol["follicle_x"].to_numpy(float)
            fy = fol["follicle_y"].to_numpy(float)
            k = np.searchsorted(fids, missing)
            a, b = fids[k - 1], fids[k]
            t = (missing - a) / (b - a)
            plans.append((side, wid, missing,
                          fx[k - 1] * (1 - t) + fx[k] * t,
                          fy[k - 1] * (1 - t) + fy[k] * t,
                          float(g["length"].median())))
            needed.update((side, int(f)) for f in missing)

    if not needed:
        return kept
    keys = pd.MultiIndex.from_arrays([combined["face_side"].to_numpy(),
                                      combined["fid"].to_numpy().astype(int)])
    comb = combined[keys.isin(needed)].copy()
    if comb.empty:
        return kept
    comb["_p"] = predict_real(comb, bundle)
    # This admission pass had three separate linear costs stacked inside one loop
    # over EVERY frame of the session, per identity, per side:
    #
    #   for f in range(fmin, fmax + 1):          # ~220k frames x ~9 ids x 2 sides
    #       lo = fol.index[fol.index < f]        # full scan of the id's frames
    #       cand = cs[(cs["fid"] == f) & ...]    # full scan of the side's table
    #
    # On a real session that is ~4M iterations each rescanning millions of rows,
    # and it is where the stage sat silently for tens of minutes. All three go
    # away without changing the result:
    #
    #   * only MISSING frames can admit anything, so iterate those, not all frames;
    #   * the bracketing frames come from searchsorted on an already-sorted index;
    #   * candidates come from a single positional groupby of `comb` by (side, fid).
    #
    # `used` stays a Python set of (fid, x, y) -- that lookup was already O(1); it
    # was pandas doing the scanning, not the set.
    comb_pos = comb.groupby(["face_side", "fid"], sort=False).indices
    c_fx = comb["follicle_x"].to_numpy(float)
    c_fy = comb["follicle_y"].to_numpy(float)
    c_p = comb["_p"].to_numpy(float)
    c_len = comb["length"].to_numpy(float)
    c_fid = comb["fid"].to_numpy()

    admitted = []                       # (position in comb, wid)
    for side, wid, missing, ex_all, ey_all, med_len in plans:
        for f, ex, ey in zip(missing, ex_all, ey_all):
            pos = comb_pos.get((side, int(f)))
            if pos is None:
                continue
            m = (c_p[pos] >= admit_threshold) & (c_len[pos] >= 0.5 * med_len)
            if not m.any():
                continue
            cand = pos[m]
            d = np.hypot(c_fx[cand] - ex, c_fy[cand] - ey)
            j = int(d.argmin())
            if d[j] > gate_px:
                continue
            q = int(cand[j])
            key = (int(c_fid[q]), round(float(c_fx[q]), 2), round(float(c_fy[q]), 2))
            if key in used:
                continue
            admitted.append((q, int(wid)))
            used.add(key)

    if admitted:
        pos = np.fromiter((p for p, _ in admitted), np.int64, len(admitted))
        wids = np.fromiter((w for _, w in admitted), np.int64, len(admitted))
        add = comb.iloc[pos].drop(columns="_p").copy()
        add["label"] = add["wid"].to_numpy()
        add["wid"] = wids
        kept = pd.concat([kept, add], ignore_index=True)
    return kept


if __name__ == "__main__":  # standalone real/noise eval, contiguous holdout
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.metrics import average_precision_score, precision_recall_curve
    D = r"E:/Thigmotaxis/whisker_active"
    comb = pd.read_parquet(f"{D}/sc013_active.parquet")
    gt = pd.read_parquet(f"{D}/sc013_active_updated_edited - backup.parquet")
    tr = np.arange(0, 3000); te = np.arange(3000, 4000)
    bundle = train_coverage(comb, gt, fids=tr)
    test = match_to_gt(comb[comb["fid"].isin(set(te))], gt)
    p = predict_real(test, bundle)
    y = test["is_real"].to_numpy(int)
    ap = average_precision_score(y, p)
    prec, rec, thr = precision_recall_curve(y, p)
    # recall at precision>=0.9
    ok = prec[:-1] >= 0.9
    rec_at_p90 = rec[:-1][ok].max() if ok.any() else 0.0
    print(f"coverage real/noise: test pos_rate={y.mean():.3f} AP={ap:.4f} "
          f"recall@prec0.9={rec_at_p90:.3f}  (n_test={len(y)})")
