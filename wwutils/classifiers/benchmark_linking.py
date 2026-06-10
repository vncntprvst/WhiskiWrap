"""Autonomous benchmark for the whisker linker and its learned add-ons.

Runs ``link_whiskers_hmm`` under a config (filters | coverage-model | identity re-rank),
scores the output against ground truth via ``eval_linking.compute_metrics`` (optionally on
a held-out contiguous fid window), and reports the headline metrics + a diff raster. This
is the objective for the autonomous tuning loop.

Split policy (one GT clip): the whisk-HMM backbone always runs on the WHOLE clip (needs
temporal context); only the learned models are train/test split, and only TEST fids are
scored. Use ``fid_range`` to score a held-out window.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd

from . import eval_linking as ev
from .hmm_link import link_whiskers_hmm

_E = r"E:/Thigmotaxis"

# clip registry: kind 'gt' = has confirmed ground truth; 'autoref' = stability only
CLIPS: Dict[str, dict] = {
    "sc013_active": dict(
        kind="gt",
        combined=f"{_E}/whisker_active/sc013_active.parquet",
        wt_dir=f"{_E}/whisker_active/WT", base_name="sc013_active",
        whiskerpad=f"{_E}/whisker_active/whiskerpad_sc013_active.json",
        gt=f"{_E}/whisker_active/sc013_active_updated_edited - backup.parquet",
        side_faces={"left": "left", "right": "right"}, n_frames=4000),
    "seg04": dict(
        kind="gt",
        combined=f"{_E}/whisker_gt_seg04/sc014_0315_001_TopCam0_seg04_493300_493931.parquet",
        wt_dir=f"{_E}/whisker_gt_seg04/WT",
        base_name="sc014_0315_001_TopCam0_seg04_493300_493931",
        whiskerpad=f"{_E}/whisker_gt_seg04/whiskerpad_sc014_0315_001_TopCam0_seg04_493300_493931.json",
        gt=f"{_E}/whisker_gt_seg04/sc014_0315_001_TopCam0_seg04_493300_493931_updated_edited.parquet",
        side_faces={"left": "left", "right": "right"}, n_frames=634),
    "longclip": dict(
        kind="autoref",
        combined=f"{_E}/whisker_longclip/sc013_longclip.parquet",
        wt_dir=f"{_E}/whisker_longclip/WT", base_name="sc013_longclip",
        whiskerpad=f"{_E}/whisker_longclip/whiskerpad_sc013_longclip.json",
        gt=None, side_faces={"left": "left", "right": "right"}, n_frames=4763),
}

HEADLINE = ["identity_accuracy", "min_coverage", "idf1", "total_id_switches",
            "miss", "false_positive", "mean_angle_mae", "extra_id_frames"]


@dataclass
class LinkerConfig:
    name: str
    coverage_mode: str = "filters"          # filters | model | hybrid
    identity_mode: str = "off"              # off | rerank
    coverage_model_path: Optional[str] = None
    identity_model_path: Optional[str] = None
    extra_kw: dict = field(default_factory=dict)


def link_clip(cfg: LinkerConfig, clip: str, out_dir: str) -> str:
    """Run the linker for ``clip`` under ``cfg``; return the predicted parquet path."""
    c = CLIPS[clip]
    os.makedirs(out_dir, exist_ok=True)
    pred = os.path.join(out_dir, f"pred_{clip}_{cfg.name}.parquet")
    link_whiskers_hmm(
        c["combined"], c["wt_dir"], c["base_name"], c["side_faces"],
        whiskerpad=c["whiskerpad"], output_path=pred,
        coverage_mode=cfg.coverage_mode, identity_mode=cfg.identity_mode,
        coverage_model_path=cfg.coverage_model_path,
        identity_model_path=cfg.identity_model_path, **cfg.extra_kw)
    return pred


def score_score(o: dict) -> float:
    # Coverage (missing whiskers) is the user's primary pain, so it is weighted at least
    # as heavily as identity: penalize raw misses explicitly and keep the id-switch weight
    # modest so a tiny idsw win can't outvote a large coverage loss.
    return (o["identity_accuracy"] + o["min_coverage"] + o["idf1"]
            - 0.003 * o["miss"] - 0.005 * o["total_id_switches"]
            - 0.05 * o["mean_angle_mae"] - 0.002 * o["extra_id_frames"])


def run_and_score(cfg: LinkerConfig, clip: str, *, out_dir: str,
                  fid_range: Optional[Tuple[int, int]] = None,
                  gate_px: float = 25.0, make_plot: bool = True) -> dict:
    """Run + score one clip. GT clips → real metrics; autoref → stability stats only."""
    c = CLIPS[clip]
    pred = link_clip(cfg, clip, out_dir)
    pred_df = pd.read_parquet(pred)
    if c["kind"] == "gt":
        gt_df = pd.read_parquet(c["gt"])
        if fid_range:
            lo, hi = fid_range
            pred_df = pred_df[(pred_df.fid >= lo) & (pred_df.fid < hi)]
            gt_df = gt_df[(gt_df.fid >= lo) & (gt_df.fid < hi)]
        m = ev.compute_metrics(pred_df, gt_df, gate_px=gate_px)
        o = {k: m["overall"].get(k) for k in HEADLINE}
        per_side = {s: v.get("identity_accuracy") for s, v in m["per_side"].items()}
        if make_plot:
            try:
                ev.plot_diff(m, os.path.join(out_dir, f"diff_{clip}_{cfg.name}.png"))
            except Exception:
                pass
        return {"clip": clip, "kind": "gt", "overall": o, "per_side": per_side,
                "score": score_score(o), "pred": pred,
                "n_pred": len(pred_df), "n_gt": len(gt_df)}
    else:
        # stability proxy (no truth): per-side det/frame, distinct wids, angle jumps
        stab = {}
        for side, g in pred_df.groupby("face_side"):
            stab[side] = dict(det_per_frame=round(len(g) / max(g.fid.nunique(), 1), 3),
                              n_wids=int(g.wid.nunique()),
                              angle_jumps=int(ev._angle_jumps(g) if hasattr(ev, "_angle_jumps") else 0))
        return {"clip": clip, "kind": "autoref", "stability": stab, "pred": pred,
                "n_pred": len(pred_df)}


def print_result(res: dict) -> None:
    if res["kind"] == "gt":
        o = res["overall"]
        print(f"[{res['clip']}] score={res['score']:.4f}  ida={o['identity_accuracy']:.4f} "
              f"min_cov={o['min_coverage']:.4f} idf1={o['idf1']:.4f} idsw={o['total_id_switches']} "
              f"miss={o['miss']} fp={o['false_positive']} angle_mae={o['mean_angle_mae']:.3f} "
              f"extra={o['extra_id_frames']} | per_side={ {s: round(v,4) for s,v in res['per_side'].items()} }")
    else:
        print(f"[{res['clip']}] (autoref) {res['stability']}")


if __name__ == "__main__":
    import argparse, warnings; warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline")
    p.add_argument("--clip", default="sc013_active")
    p.add_argument("--coverage", default="filters")
    p.add_argument("--identity", default="off")
    p.add_argument("--coverage-model", default=None)
    p.add_argument("--identity-model", default=None)
    p.add_argument("--fid-lo", type=int, default=None)
    p.add_argument("--fid-hi", type=int, default=None)
    p.add_argument("--out", default=f"{_E}/_autotune")
    a = p.parse_args()
    cfg = LinkerConfig(a.config, a.coverage, a.identity, a.coverage_model, a.identity_model)
    fr = (a.fid_lo, a.fid_hi) if a.fid_lo is not None else None
    res = run_and_score(cfg, a.clip, out_dir=a.out, fid_range=fr)
    print_result(res)
