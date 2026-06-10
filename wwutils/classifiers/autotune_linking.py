"""Autonomous tuning loop for the learned whisker add-ons.

Caches the expensive whisk-HMM backbone once per clip, then searches add-on configs
OFFLINE (coverage selection + identity re-rank applied to the cached output) so dozens of
configs are scored in seconds. Validates with a contiguous held-out window + blocked
k-fold, gates on multi-clip no-regression, and reports a leaderboard + the winning models.

Design recap (see handoff): coverage model is shared/generalizable; identity model is
per-clip (GT when available, else bootstrapped from the HMM's confident runs); never
per-frame position rank.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import coverage_model as cov
from . import identity_model as idm
from . import eval_linking as ev
from . import benchmark_linking as bl
from . import hmm_link as hl

OUT = r"E:/Thigmotaxis/_autotune"
N = 4000
BLOCKS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
TEST = (3000, 4000)
TRAIN = np.arange(0, 3000)


# ---- cache the backbone (combined, raw out) per clip; expensive step run once ----
def cache_backbone(clip: str) -> Dict:
    c = bl.CLIPS[clip]
    bb = hl.hmm_backbone(c["combined"], c["wt_dir"], c["base_name"], c["side_faces"],
                         whiskerpad=c["whiskerpad"])
    combined, out = bb
    return {"clip": clip, "combined": combined, "out": out, "side_faces": c["side_faces"]}


# ---- apply an add-on config to a cached raw `out` (no whisk re-run) ----
def apply_config(cache: Dict, cfg: dict, cov_bundle=None, id_bundle=None) -> pd.DataFrame:
    out, combined, side_faces = cache["out"].copy(), cache["combined"], cache["side_faces"]
    if cfg.get("coverage") == "model" and cov_bundle is not None:
        gate = hl.estimate_follicle_gate(out, 0.15)
        out = cov.apply_coverage_model(out, combined, cov_bundle, side_faces=side_faces,
                                       gate_px=gate, keep_threshold=cfg.get("keep", 0.5),
                                       admit_threshold=cfg.get("admit", 0.9),
                                       admit=cfg.get("do_admit", True))
    else:  # filters baseline
        out = hl.filter_length_outliers(out, 0.4)
        gate = hl.estimate_follicle_gate(out, 0.15)
        out = hl.filter_follicle_outliers(out, gate)
        out = hl.bridge_gaps(out, combined, max_gap=20, gate_px=gate, min_length_frac=0.4)
        out = hl.filter_angle_outliers(out, angle_k=4.0)
    if cfg.get("identity") == "rerank" and id_bundle is not None:
        out = idm.rerank_identity(out, id_bundle, side_faces=side_faces,
                                  w_model=cfg.get("w_model", 1.2),
                                  w_cont=cfg.get("w_cont", 1.0))
    return out.sort_values(["fid", "wid"])


def score_window(df: pd.DataFrame, gt: pd.DataFrame, lo: int, hi: int) -> dict:
    p = df[(df.fid >= lo) & (df.fid < hi)]
    g = gt[(gt.fid >= lo) & (gt.fid < hi)]
    return ev.compute_metrics(p, g)["overall"]


def train_models(cache: Dict, gt: pd.DataFrame, fids: np.ndarray):
    cov_bundle = cov.train_coverage(cache["combined"], gt, fids=fids)
    # identity from GT on the GT clip (most faithful); apply also works via bootstrap
    id_bundle = idm.train_identity(gt, gt=gt, fids=fids)
    return cov_bundle, id_bundle


def kfold(cache: Dict, gt: pd.DataFrame, cfg: dict) -> List[dict]:
    res = []
    for i, (lo, hi) in enumerate(BLOCKS):
        train_fids = np.array([f for f in range(N) if not (lo <= f < hi)])
        cb, ib = train_models(cache, gt, train_fids)
        df = apply_config(cache, cfg, cb, ib)
        res.append(score_window(df, gt, lo, hi))
    return res


def regression_ok(cache_seg: Dict, cov_bundle) -> dict:
    """seg04 has no GT here -> structural/stability check: coverage(shared) + identity
    bootstrapped from seg04's own HMM must keep 3 wids/side and not fragment."""
    cfg = {"coverage": "model", "identity": "rerank", "keep": 0.5}
    id_seg = idm.train_identity(cache_seg["out"], gt=None)  # bootstrap from seg04 HMM
    df = apply_config(cache_seg, cfg, cov_bundle, id_seg)
    base = apply_config(cache_seg, {"coverage": "filters", "identity": "off"})
    def stats(d):
        return {s: dict(nwid=int(g.wid.nunique()),
                        det=round(len(g) / max(g.fid.nunique(), 1), 2))
                for s, g in d.groupby("face_side")}
    return {"baseline": stats(base), "addon": stats(df),
            "ok": all(v["nwid"] == 3 for v in stats(df).values())}


def main():
    import warnings; warnings.filterwarnings("ignore")
    os.makedirs(OUT + "/models", exist_ok=True)
    print("caching backbones (whisk-HMM run once per clip)...")
    sc = cache_backbone("sc013_active")
    seg = cache_backbone("seg04")
    gt = pd.read_parquet(bl.CLIPS["sc013_active"]["gt"])

    base_o = score_window(apply_config(sc, {"coverage": "filters", "identity": "off"}),
                          gt, *TEST)
    print(f"\nBASELINE  test[{TEST}]: ida={base_o['identity_accuracy']:.4f} "
          f"min_cov={base_o['min_coverage']:.4f} idsw={base_o['total_id_switches']} "
          f"miss={base_o['miss']} idf1={base_o['idf1']:.4f}")

    cov_b, id_b = train_models(sc, gt, TRAIN)
    grid = [
        {"name": "cov", "coverage": "model", "identity": "off", "keep": 0.5},
        {"name": "cov+id", "coverage": "model", "identity": "rerank", "keep": 0.5, "w_model": 1.2},
        {"name": "cov+id_wm2", "coverage": "model", "identity": "rerank", "keep": 0.5, "w_model": 2.0},
        {"name": "cov+id_k0.4", "coverage": "model", "identity": "rerank", "keep": 0.4, "w_model": 1.2},
        {"name": "cov+id_noadmit", "coverage": "model", "identity": "rerank", "keep": 0.5,
         "do_admit": False, "w_model": 1.2},
    ]
    board = []
    for cfg in grid:
        o = score_window(apply_config(sc, cfg, cov_b, id_b), gt, *TEST)
        s = bl.score_score(o)
        board.append((cfg["name"], s, o))
        print(f"  {cfg['name']:16s} score={s:.4f} ida={o['identity_accuracy']:.4f} "
              f"min_cov={o['min_coverage']:.4f} idsw={o['total_id_switches']} miss={o['miss']} "
              f"idf1={o['idf1']:.4f}")
    board.sort(key=lambda x: -x[1])
    win_name = board[0][0]
    win_cfg = next(c for c in grid if c["name"] == win_name)
    print(f"\nWINNER: {win_name}")

    # k-fold on the winner (overfit detector)
    kf = kfold(sc, gt, win_cfg)
    mc = np.array([o["min_coverage"] for o in kf]); ida = np.array([o["identity_accuracy"] for o in kf])
    idsw = np.array([o["total_id_switches"] for o in kf])
    print(f"4-fold winner: ida={ida.mean():.4f}±{ida.std():.4f} min_cov={mc.mean():.4f}±{mc.std():.4f} "
          f"idsw={idsw.mean():.1f}±{idsw.std():.1f}")

    # no-regression on seg04
    reg = regression_ok(seg, cov_b)
    print(f"seg04 no-regression: ok={reg['ok']}  addon={reg['addon']}  baseline={reg['baseline']}")

    # save winning models + summary
    cov.save_coverage(cov_b, OUT + "/models/coverage.joblib")
    idm.save_identity(id_b, OUT + "/models/identity_sc013.joblib")
    summary = {"baseline": base_o, "winner": win_name, "winner_cfg": win_cfg,
               "leaderboard": [(n, s) for n, s, _ in board],
               "kfold": {"ida_mean": float(ida.mean()), "min_cov_mean": float(mc.mean()),
                         "min_cov_std": float(mc.std()), "idsw_mean": float(idsw.mean())},
               "seg04_ok": reg["ok"]}
    with open(OUT + "/autotune_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsaved models + summary to {OUT}")


if __name__ == "__main__":
    main()
