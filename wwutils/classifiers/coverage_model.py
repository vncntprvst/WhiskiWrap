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


def _features(df: pd.DataFrame, k: int = 16) -> pd.DataFrame:
    feats = dF.detection_features(df, k=k)
    return feats[dF.COVERAGE_FEATURES]


def train_coverage(combined: pd.DataFrame, gt: pd.DataFrame, *,
                   fids: Optional[np.ndarray] = None, k: int = 16,
                   gate_px: float = 12.0) -> Dict:
    """Train the real/noise classifier. Labels come from matching combined->GT.

    ``fids`` restricts TRAINING frames (held-out frames are excluded to avoid leakage).
    Returns a bundle dict with the fitted model and metadata.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required for the coverage model")
    train = combined if fids is None else combined[combined["fid"].isin(set(fids))]
    labeled = match_to_gt(train, gt, gate_px=gate_px)
    X = _features(labeled, k=k)
    y = labeled["is_real"].to_numpy(int)
    # balance the ~8% positive class
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1,
                                         max_iter=300, l2_regularization=1.0,
                                         early_stopping=True, random_state=0)
    clf.fit(X, y, sample_weight=w)
    return {"model": clf, "features": dF.COVERAGE_FEATURES, "k": k,
            "n_train": len(y), "pos_rate": float(y.mean())}


def train_coverage_multi(pairs, *, k: int = 16, gate_px: float = 12.0) -> Dict:
    """Train one production coverage model from several clips.

    ``pairs`` = list of (combined_df, gt_df). Pooling multiple animals makes the universal
    real-vs-noise boundary more robust. Labels come from matching each clip's combined->GT.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required for the coverage model")
    Xs, ys = [], []
    for comb, gt in pairs:
        labeled = match_to_gt(comb, gt, gate_px=gate_px)
        Xs.append(_features(labeled, k=k))
        ys.append(labeled["is_real"].to_numpy(int))
    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=300,
                                         l2_regularization=1.0, early_stopping=True,
                                         random_state=0)
    clf.fit(X, y, sample_weight=w)
    return {"model": clf, "features": dF.COVERAGE_FEATURES, "k": k,
            "n_train": len(y), "pos_rate": float(y.mean()), "n_clips": len(pairs)}


def predict_real(df: pd.DataFrame, bundle: Dict) -> np.ndarray:
    """Return p(real) for each row of ``df``."""
    X = _features(df, k=bundle.get("k", 16))
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

    # index combined p(real) once; mark which combined rows are already used
    comb = combined.copy()
    comb["_p"] = predict_real(comb, bundle)
    used = set(zip(kept["fid"].astype(int), kept["follicle_x"].round(2),
                   kept["follicle_y"].round(2)))
    new_rows = []
    for side in kept["face_side"].unique():
        cs = comb[comb["face_side"] == side]
        for wid, g in kept[kept["face_side"] == side].groupby("wid"):
            g = g.sort_values("fid")
            fol = g.drop_duplicates("fid").set_index("fid")[["follicle_x", "follicle_y"]]
            present = set(g["fid"].tolist())
            fmin, fmax = int(g["fid"].iloc[0]), int(g["fid"].iloc[-1])
            med_len = float(g["length"].median())
            for f in range(fmin, fmax + 1):
                if f in present:
                    continue
                # interpolate expected follicle from bracketing present frames
                lo = fol.index[fol.index < f]
                hi = fol.index[fol.index > f]
                if len(lo) == 0 or len(hi) == 0:
                    continue
                a, b = lo[-1], hi[0]
                t = (f - a) / (b - a)
                ex = fol.loc[a, "follicle_x"] * (1 - t) + fol.loc[b, "follicle_x"] * t
                ey = fol.loc[a, "follicle_y"] * (1 - t) + fol.loc[b, "follicle_y"] * t
                cand = cs[(cs["fid"] == f) & (cs["_p"] >= admit_threshold)
                          & (cs["length"] >= 0.5 * med_len)]
                if cand.empty:
                    continue
                d = np.hypot(cand["follicle_x"] - ex, cand["follicle_y"] - ey)
                if d.min() > gate_px:
                    continue
                row = cand.loc[d.idxmin()].copy()
                key = (int(row["fid"]), round(row["follicle_x"], 2), round(row["follicle_y"], 2))
                if key in used:
                    continue
                row["label"] = row["wid"]
                row["wid"] = int(wid)
                new_rows.append(row.drop(labels="_p"))
                used.add(key)
    if new_rows:
        kept = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
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
