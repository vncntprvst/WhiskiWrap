"""IDENTITY model: a per-session, conservative learned re-ranker (close-whisker fix).

Identity is session-specific (which whisker is "id4" depends on this session's pad), so a
frozen classifier cannot transfer. This trains a tiny per-side model on that session's
labels (GT when available, else the whisk-HMM's confident runs as pseudo-labels) keyed on
the stable signatures the position-only linker ignores -- chiefly LENGTH and base shape --
and uses it only as a re-ranker.

It NEVER re-derives identity from per-frame position rank (that flickers). It resolves each
side by a forward one-whisker-per-frame Hungarian assignment whose cost blends: (i) a strong
prior to keep the whisk-HMM identity, (ii) follicle continuity from the previous assigned
frame, and (iii) the learned class probability. The model only overrides the HMM when its
evidence is strong and continuity allows it -- so it cannot do worse than the HMM by much,
and the benchmark gate enforces no regression.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import detection_features as dF
from .gt_labels import match_to_gt

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False


def _feat(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return dF.detection_features(df, k=k)[dF.IDENTITY_FEATURES]


def train_identity(linked: pd.DataFrame, *, gt: Optional[pd.DataFrame] = None,
                   fids: Optional[np.ndarray] = None, k: int = 16,
                   min_run: int = 5) -> Dict:
    """Train one per-side identity classifier.

    Labels: if ``gt`` given, match linked->GT to get the true wid (training/eval clip);
    else bootstrap from ``linked['wid']`` keeping only identities on long stable runs
    (pseudo-labels from the whisk-HMM). ``fids`` restricts training frames.
    """
    if not _SKLEARN:
        raise ImportError("scikit-learn required for the identity model")
    df = linked if fids is None else linked[linked["fid"].isin(set(fids))]
    if gt is not None:
        lab = match_to_gt(df, gt)
        lab = lab[lab["is_real"]].copy()
        lab["y"] = lab["gt_wid"].astype(int)
    else:
        lab = df.copy()
        # keep only long contiguous runs per (side,wid) as confident pseudo-labels
        keep = []
        for (_, _), g in lab.groupby(["face_side", "wid"]):
            g = g.sort_values("fid")
            runs = np.split(g, np.where(np.diff(g["fid"].values) != 1)[0] + 1)
            for r in runs:
                if len(r) >= min_run:
                    keep.append(r)
        lab = pd.concat(keep) if keep else lab
        lab["y"] = lab["wid"].astype(int)
    models = {}
    for side, g in lab.groupby("face_side"):
        classes = sorted(g["y"].unique())
        if len(classes) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1,
                                             max_iter=300, l2_regularization=1.0,
                                             early_stopping=True, random_state=0)
        clf.fit(_feat(g, k), g["y"].to_numpy(int))
        models[side] = (clf, list(clf.classes_))
    return {"models": models, "features": dF.IDENTITY_FEATURES, "k": k}


def _assoc_features(fol, tip, ang, length, rank,
                    last_fol, last_tip, last_ang, last_len, last_rank, gap, med):
    """Feature row for the learned association model, in its training order.

    Displacements are divided by `med`, the typical whisker displacement between
    this pair of frames, which is what makes the model portable across frame rates
    -- it was trained at 200 fps and is used here at 500.
    """
    return [
        float(np.hypot(*(fol - last_fol))) / med,
        abs(((ang - last_ang + 180) % 360) - 180) if (ang is not None
                                                      and last_ang is not None) else 0.0,
        abs(length - last_len) / max(last_len, 1.0) if last_len else 0.0,
        float(np.hypot(*(tip - last_tip))) / med if (tip is not None
                                                     and last_tip is not None) else 0.0,
        abs(rank - last_rank) if (rank is not None and last_rank is not None) else 0.0,
        float(gap),
    ]


def rerank_identity(out: pd.DataFrame, bundle: Dict, *, side_faces=None,
                    w_prior: float = 1.0, w_cont: float = 1.0, w_model: float = 1.2,
                    cont_scale: float = 30.0,
                    w_angle: float = 1.0, angle_scale: float = 10.0,
                    assoc_bundle=None, w_assoc: float = 2.0) -> pd.DataFrame:
    """Forward one-per-frame re-assignment per side (no flicker). Schema unchanged.

    Continuity is measured on BOTH follicle position and angle. Follicle alone
    fails exactly where it is needed most: when whisk forks at a whisker crossing
    it emits two segments sharing one base, so their follicles are 0-1 px apart and
    the positional term contributes nothing to either candidate. Measured on a poke
    excerpt, the two distal parts then sat at a stable ~66 deg and ~85 deg while
    the assignment between them alternated frame to frame -- the ~19 deg jumps that
    look like identity switches. Angle separates those two candidates cleanly when
    position cannot.
    """
    if out.empty or not bundle.get("models"):
        return out
    k = bundle.get("k", 16)
    res = out.copy()
    for side, (clf, classes) in bundle["models"].items():
        s = res[res["face_side"] == side]
        if s.empty:
            continue
        probs = clf.predict_proba(_feat(s, k))            # [n, C] aligned to s
        prob_by_idx = {idx: probs[i] for i, idx in enumerate(s.index)}
        cls_pos = {c: j for j, c in enumerate(classes)}
        last_fol: Dict[int, Optional[np.ndarray]] = {c: None for c in classes}
        last_ang: Dict[int, Optional[float]] = {c: None for c in classes}
        last_tip: Dict[int, Optional[np.ndarray]] = {c: None for c in classes}
        last_len: Dict[int, float] = {c: 0.0 for c in classes}
        last_rank: Dict[int, Optional[int]] = {c: None for c in classes}
        last_fid: Dict[int, Optional[int]] = {c: None for c in classes}
        has_tip = "tip_x" in res.columns and "tip_y" in res.columns
        has_len = "length" in res.columns
        aclf = assoc_bundle["model"] if assoc_bundle else None
        has_angle = "angle" in res.columns
        # Group once. `s[s["fid"] == fid]` inside the loop builds a boolean mask
        # over the whole side for every frame, which is O(rows x frames): on a
        # 123k-frame session with 1.75M rows per side that is ~2e11 comparisons and
        # it turned linking into an 8-hour serial stage. groupby is one pass.
        by_fid = {int(f): g for f, g in s.groupby("fid", sort=True)}
        for fid in sorted(by_fid):
            fr = by_fid[fid]
            idxs = list(fr.index)
            C = np.zeros((len(idxs), len(classes)))
            # One predict_proba for the whole frame rather than one per candidate
            # pair. The per-pair version made a 5000-frame clip take longer than a
            # ten-minute budget -- with n detections and k identities it is n*k
            # calls per frame, each with its own sklearn overhead.
            assoc_rows, assoc_at = [], []
            # Typical displacement for THIS frame pair, used to normalise the
            # association features. Computed from each detection's nearest previous
            # identity, which does not depend on the assignment being made -- using
            # the assignment itself would be circular.
            med = 1.0
            if aclf is not None:
                dmins = []
                for idx in idxs:
                    f0 = np.array([res.at[idx, "follicle_x"],
                                   res.at[idx, "follicle_y"]], float)
                    ds = [float(np.hypot(*(f0 - last_fol[c])))
                          for c in classes if last_fol[c] is not None]
                    if ds:
                        dmins.append(min(ds))
                if dmins:
                    med = max(float(np.median(dmins)), 1e-3)
                cur_rank = {idx: r for r, idx in enumerate(
                    sorted(idxs, key=lambda i: float(res.at[i, "follicle_y"])))}
            for r, idx in enumerate(idxs):
                p = prob_by_idx[idx]
                cur = int(res.at[idx, "wid"])
                fol = np.array([res.at[idx, "follicle_x"], res.at[idx, "follicle_y"]], float)
                ang = float(res.at[idx, "angle"]) if has_angle else None
                for jc, c in enumerate(classes):
                    prior = 0.0 if cur == c else 1.0
                    model = 1.0 - float(p[cls_pos[c]])
                    if last_fol[c] is None:
                        cont = 0.0
                    else:
                        cont = min(np.hypot(*(fol - last_fol[c])) / cont_scale, 2.0)
                    # Angle continuity. Capped like the positional term so a single
                    # large excursion cannot dominate the assignment, but able to
                    # separate two candidates that share a follicle.
                    if ang is None or last_ang[c] is None:
                        acont = 0.0
                    else:
                        acont = min(abs(ang - last_ang[c]) / angle_scale, 2.0)
                    C[r, jc] = (w_prior * prior + w_model * model
                                + w_cont * cont + w_angle * acont)
                    if aclf is not None and last_fol[c] is not None:
                        tipv = (np.array([res.at[idx, "tip_x"], res.at[idx, "tip_y"]],
                                         float) if has_tip else None)
                        lenv = float(res.at[idx, "length"]) if has_len else 0.0
                        gap = (fid - last_fid[c] - 1) if last_fid[c] is not None else 0
                        assoc_rows.append(_assoc_features(
                            fol, tipv, ang, lenv, cur_rank.get(idx),
                            last_fol[c], last_tip[c], last_ang[c], last_len[c],
                            last_rank[c], max(gap, 0), med))
                        assoc_at.append((r, jc))
            if assoc_rows:
                psame = aclf.predict_proba(np.asarray(assoc_rows, float))[:, 1]
                for (r, jc), ps in zip(assoc_at, psame):
                    C[r, jc] += w_assoc * (1.0 - float(ps))
            ri, ci = linear_sum_assignment(C)
            for r, jc in zip(ri, ci):
                idx = idxs[r]; c = classes[jc]
                res.at[idx, "wid"] = c
                last_fol[c] = np.array([res.at[idx, "follicle_x"],
                                        res.at[idx, "follicle_y"]], float)
                if has_angle:
                    last_ang[c] = float(res.at[idx, "angle"])
                if aclf is not None:
                    if has_tip:
                        last_tip[c] = np.array([res.at[idx, "tip_x"],
                                                res.at[idx, "tip_y"]], float)
                    if has_len:
                        last_len[c] = float(res.at[idx, "length"])
                    last_rank[c] = cur_rank.get(idx)
                    last_fid[c] = int(fid)
    return res


def save_identity(bundle, path):
    import joblib; joblib.dump(bundle, path)


if __name__ == "__main__":  # standalone identity eval, contiguous holdout
    import os, warnings; warnings.filterwarnings("ignore")
    from . import benchmark_linking as bl
    from . import eval_linking as ev
    OUT = r"E:/Thigmotaxis/_autotune"
    # train on the baseline-linked output of [0,3000), score rerank on [3000,4000)
    base_pred = os.path.join(OUT, "pred_sc013_active_baseline.parquet")
    linked = pd.read_parquet(base_pred)
    gt = pd.read_parquet(bl.CLIPS["sc013_active"]["gt"])
    b = train_identity(linked, gt=gt, fids=np.arange(0, 3000))
    print("identity models per side:", {s: m[1] for s, m in b["models"].items()})
    te = linked[(linked.fid >= 3000) & (linked.fid < 4000)]
    gte = gt[(gt.fid >= 3000) & (gt.fid < 4000)]
    before = ev.compute_metrics(te, gte)["overall"]
    re = rerank_identity(te, b)
    after = ev.compute_metrics(re, gte)["overall"]
    for tag, o in (("before", before), ("after ", after)):
        print(f"{tag}: ida={o['identity_accuracy']:.4f} idsw={o['total_id_switches']} "
              f"idf1={o['idf1']:.4f} mismatch={o['mismatch']}")
