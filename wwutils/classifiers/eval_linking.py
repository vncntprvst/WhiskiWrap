"""Quantitative evaluation of whisker linking quality.

Compares a *predicted* whisker-tracking parquet against a manually corrected
*ground-truth* parquet and reports how well whisker identity (``wid``) is
preserved across frames. Identity is the headline concern: each persistent
``wid`` should refer to the same physical whisker for the whole video so that
per-whisker kinematic traces (especially angle) are correct.

The two files do not share a row order and their integer ``wid`` values do not
match, so evaluation proceeds in two stages:

1. Per (frame, face_side), match predicted detections to ground-truth
   detections by follicle geometry (optimal assignment with a distance gate).
2. Globally align predicted ``wid`` -> ground-truth ``wid`` one-to-one from how
   often they co-occur, so identity switches are penalised rather than hidden.

Metrics are then computed per face_side (sides never mix) and aggregated.

This module is intentionally dependency-light: it imports only pandas / numpy /
scipy at module load. matplotlib is imported lazily inside the plotting helper
so the metric path stays headless.

CLI:
    python eval_linking.py --pred PRED.parquet --gt GT.parquet \
        [--angle-jump-thresh 20] [--diff-plot OUT.png] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Sentinel ground-truth id for predicted whiskers that match no real whisker.
NO_GT = -1


@dataclass
class Thresholds:
    """Pass/fail thresholds for a regression check (defaults are lenient)."""
    min_identity_accuracy: float = 0.95
    min_idf1: float = 0.95
    max_id_switches: int = 4
    min_coverage: float = 0.9
    max_angle_mae: float = 5.0          # degrees
    max_extra_id_frames: int = 10       # total frames occupied by spurious ids


def _match_one_frame(pred: pd.DataFrame, gt: pd.DataFrame, gate_px: float,
                     angle_weight: float) -> List[Tuple[int, int]]:
    """Optimally match predicted rows to gt rows in a single (frame, side).

    Returns a list of ``(pred_pos, gt_pos)`` integer-position pairs whose
    follicle distance is within ``gate_px``. Unmatched rows are simply absent.
    """
    if len(pred) == 0 or len(gt) == 0:
        return []
    px = pred[["follicle_x", "follicle_y"]].to_numpy(float)
    gx = gt[["follicle_x", "follicle_y"]].to_numpy(float)
    # Euclidean follicle distance, optionally nudged by angle similarity.
    dist = np.sqrt(((px[:, None, :] - gx[None, :, :]) ** 2).sum(-1))
    cost = dist.copy()
    if angle_weight:
        pa = pred["angle"].to_numpy(float)
        ga = gt["angle"].to_numpy(float)
        cost = cost + angle_weight * np.abs(pa[:, None] - ga[None, :])
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if dist[r, c] <= gate_px]


def build_match_table(pred_df: pd.DataFrame, gt_df: pd.DataFrame, *,
                      gate_px: float = 25.0, angle_weight: float = 0.0,
                      wid_col: str = "wid") -> pd.DataFrame:
    """Per (fid, face_side) geometric matching of pred to gt detections.

    Returns a long table with one row per ground-truth detection plus one row
    per *unmatched* predicted detection, with columns:
        fid, face_side, gt_wid, pred_wid, matched, error_type,
        follicle_x, follicle_y, gt_angle, pred_angle
    ``error_type`` is one of: 'ok' (matched), 'miss' (gt with no pred),
    'fp' (pred with no gt).
    """
    records: List[dict] = []
    sides = sorted(set(gt_df["face_side"].unique()) | set(pred_df["face_side"].unique()))
    for side in sides:
        p_side = pred_df[pred_df["face_side"] == side]
        g_side = gt_df[gt_df["face_side"] == side]
        fids = sorted(set(g_side["fid"].unique()) | set(p_side["fid"].unique()))
        for fid in fids:
            p = p_side[p_side["fid"] == fid]
            g = g_side[g_side["fid"] == fid]
            matches = _match_one_frame(p, g, gate_px, angle_weight)
            matched_p = {m[0] for m in matches}
            matched_g = {m[1] for m in matches}
            for pp, gg in matches:
                prow, grow = p.iloc[pp], g.iloc[gg]
                records.append(dict(
                    fid=fid, face_side=side,
                    gt_wid=int(grow[wid_col]), pred_wid=int(prow[wid_col]),
                    matched=True, error_type="ok",
                    follicle_x=float(grow["follicle_x"]), follicle_y=float(grow["follicle_y"]),
                    gt_angle=float(grow["angle"]), pred_angle=float(prow["angle"]),
                ))
            for gi in range(len(g)):
                if gi in matched_g:
                    continue
                grow = g.iloc[gi]
                records.append(dict(
                    fid=fid, face_side=side,
                    gt_wid=int(grow[wid_col]), pred_wid=NO_GT,
                    matched=False, error_type="miss",
                    follicle_x=float(grow["follicle_x"]), follicle_y=float(grow["follicle_y"]),
                    gt_angle=float(grow["angle"]), pred_angle=np.nan,
                ))
            for pi in range(len(p)):
                if pi in matched_p:
                    continue
                prow = p.iloc[pi]
                records.append(dict(
                    fid=fid, face_side=side,
                    gt_wid=NO_GT, pred_wid=int(prow[wid_col]),
                    matched=False, error_type="fp",
                    follicle_x=float(prow["follicle_x"]), follicle_y=float(prow["follicle_y"]),
                    gt_angle=np.nan, pred_angle=float(prow["angle"]),
                ))
    cols = ["fid", "face_side", "gt_wid", "pred_wid", "matched", "error_type",
            "follicle_x", "follicle_y", "gt_angle", "pred_angle"]
    return pd.DataFrame.from_records(records, columns=cols)


def align_wids(match_table: pd.DataFrame) -> Dict[int, int]:
    """One-to-one global map predicted_wid -> gt_wid (per side, merged).

    Built from the contingency of how often each predicted wid is geometrically
    matched to each gt wid, solved with Hungarian. Predicted wids with no
    profitable match are omitted from the map (callers treat them as spurious).
    Because sides have disjoint wid ranges, solving globally is equivalent to
    solving per side but simpler.
    """
    ok = match_table[(match_table["error_type"] == "ok")]
    if ok.empty:
        return {}
    contingency = (ok.groupby(["pred_wid", "gt_wid"]).size()
                   .reset_index(name="n"))
    pred_ids = sorted(contingency["pred_wid"].unique())
    gt_ids = sorted(contingency["gt_wid"].unique())
    pidx = {p: i for i, p in enumerate(pred_ids)}
    gidx = {g: j for j, g in enumerate(gt_ids)}
    M = np.zeros((len(pred_ids), len(gt_ids)), dtype=float)
    for _, r in contingency.iterrows():
        M[pidx[int(r["pred_wid"])], gidx[int(r["gt_wid"])]] = r["n"]
    rows, cols = linear_sum_assignment(-M)
    mapping: Dict[int, int] = {}
    for r, c in zip(rows, cols):
        if M[r, c] > 0:  # only keep matches that actually co-occur
            mapping[pred_ids[r]] = gt_ids[c]
    return mapping


def _contiguous_runs(fids: np.ndarray) -> int:
    """Number of maximal runs of consecutive frame ids (gap = fid jump > 1)."""
    if len(fids) == 0:
        return 0
    fids = np.sort(fids)
    return int(1 + np.count_nonzero(np.diff(fids) > 1))


def _angle_jumps(df: pd.DataFrame, wid_col: str, thresh: float) -> int:
    """Count consecutive-frame angle changes exceeding ``thresh`` per whisker."""
    n = 0
    for _, g in df.groupby(wid_col):
        g = g.sort_values("fid")
        fid = g["fid"].to_numpy()
        ang = g["angle"].to_numpy(float)
        consecutive = np.diff(fid) == 1
        jumps = np.abs(np.diff(ang)) > thresh
        n += int(np.count_nonzero(consecutive & jumps))
    return n


def compute_metrics(pred_df: pd.DataFrame, gt_df: pd.DataFrame, *,
                    gate_px: float = 25.0, angle_jump_thresh: float = 20.0,
                    wid_col: str = "wid") -> dict:
    """Compute the full suite of linking-quality metrics.

    Returns a nested dict: ``overall`` aggregate metrics, ``per_side`` and
    ``per_gt_whisker`` breakdowns, plus the ``wid_map`` and ``extra_ids``.
    """
    mt = build_match_table(pred_df, gt_df, gate_px=gate_px, wid_col=wid_col)
    wid_map = align_wids(mt)  # pred_wid -> gt_wid
    inv_map: Dict[int, int] = {g: p for p, g in wid_map.items()}  # gt_wid -> pred_wid

    # Identity correctness for matched gt detections.
    ok = mt[mt["error_type"] == "ok"].copy()
    ok["pred_gt"] = ok["pred_wid"].map(wid_map)  # what gt the pred claims
    ok["correct"] = ok["pred_gt"] == ok["gt_wid"]

    total_gt = int((mt["gt_wid"] != NO_GT).sum())
    n_miss = int((mt["error_type"] == "miss").sum())
    n_fp = int((mt["error_type"] == "fp").sum())
    n_matched = int(len(ok))
    n_correct = int(ok["correct"].sum())
    n_mismatch = n_matched - n_correct

    identity_accuracy = n_correct / total_gt if total_gt else 1.0
    mota = 1.0 - (n_miss + n_fp + n_mismatch) / total_gt if total_gt else 1.0

    # IDF1 from the global one-to-one identity match.
    id_tp = n_correct
    id_fn = total_gt - id_tp
    total_pred = int((mt["pred_wid"] != NO_GT).sum())
    id_fp = total_pred - id_tp
    idf1 = (2 * id_tp / (2 * id_tp + id_fp + id_fn)) if (2 * id_tp + id_fp + id_fn) else 1.0

    # Per ground-truth whisker breakdown.
    per_gt: Dict[str, dict] = {}
    total_idsw = 0
    for gt_wid, g in mt[mt["gt_wid"] != NO_GT].groupby("gt_wid"):
        g = g.sort_values("fid")
        gt_frames = int(g["fid"].nunique())
        gok = ok[ok["gt_wid"] == gt_wid].sort_values("fid")
        correct_frames = int(gok["correct"].sum())
        # ID switches: changes in the matched predicted wid across present frames.
        present = gok[gok["matched"]]
        pred_seq = present["pred_wid"].to_numpy()
        idsw = int(np.count_nonzero(np.diff(pred_seq) != 0)) if len(pred_seq) > 1 else 0
        total_idsw += idsw
        # Fragmentation: contiguous runs where identity is correct.
        frag = _contiguous_runs(gok[gok["correct"]]["fid"].to_numpy())
        # Angle error vs the predicted whisker mapped to this gt whisker.
        mae = rmse = np.nan
        if gt_wid in inv_map:
            p_wid = inv_map[gt_wid]
            p_tr = pred_df[(pred_df[wid_col] == p_wid)][["fid", "angle"]]
            g_tr = gt_df[(gt_df[wid_col] == gt_wid)][["fid", "angle"]]
            merged = p_tr.merge(g_tr, on="fid", suffixes=("_pred", "_gt"))
            if len(merged):
                diff = (merged["angle_pred"] - merged["angle_gt"]).to_numpy(float)
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff ** 2)))
        per_gt[str(int(gt_wid))] = dict(
            gt_frames=gt_frames, correct_frames=correct_frames,
            coverage=correct_frames / gt_frames if gt_frames else 0.0,
            id_switches=idsw, fragmentation=frag,
            angle_mae=mae, angle_rmse=rmse,
            mapped_pred_wid=int(inv_map.get(gt_wid, NO_GT)),
        )

    # Spurious predicted ids (mapped to no gt whisker).
    extra_ids = sorted(set(int(w) for w in pred_df[wid_col].unique()) - set(wid_map.keys()))
    extra_frames = int(pred_df[pred_df[wid_col].isin(extra_ids)]["fid"].nunique()) if extra_ids else 0
    extra_rows = int(pred_df[pred_df[wid_col].isin(extra_ids)].shape[0]) if extra_ids else 0

    # Angle jumps (predicted vs the gt baseline).
    pred_jumps = _angle_jumps(pred_df[pred_df[wid_col].isin(wid_map.keys())], wid_col, angle_jump_thresh)
    gt_jumps = _angle_jumps(gt_df, wid_col, angle_jump_thresh)

    # Per-side aggregate.
    per_side: Dict[str, dict] = {}
    for side, s in mt.groupby("face_side"):
        s_ok = ok[ok["face_side"] == side]
        s_total_gt = int((s["gt_wid"] != NO_GT).sum())
        per_side[side] = dict(
            identity_accuracy=(int(s_ok["correct"].sum()) / s_total_gt) if s_total_gt else 1.0,
            gt_detections=s_total_gt,
            miss=int((s["error_type"] == "miss").sum()),
            fp=int((s["error_type"] == "fp").sum()),
        )

    overall_mae = float(np.nanmean([v["angle_mae"] for v in per_gt.values()])) if per_gt else np.nan
    worst_mae = float(np.nanmax([v["angle_mae"] for v in per_gt.values()])) if per_gt else np.nan
    min_cov = float(np.min([v["coverage"] for v in per_gt.values()])) if per_gt else 1.0

    return dict(
        overall=dict(
            identity_accuracy=identity_accuracy,
            mota=mota, idf1=idf1,
            total_id_switches=total_idsw,
            total_gt_detections=total_gt,
            miss=n_miss, false_positive=n_fp, mismatch=n_mismatch,
            mean_angle_mae=overall_mae, worst_angle_mae=worst_mae,
            min_coverage=min_cov,
            pred_angle_jumps=pred_jumps, gt_angle_jumps=gt_jumps,
            extra_ids=extra_ids, extra_id_frames=extra_frames, extra_id_rows=extra_rows,
        ),
        per_side=per_side,
        per_gt_whisker=per_gt,
        wid_map={int(k): int(v) for k, v in wid_map.items()},
        _match_table=mt,  # retained for plotting; stripped before json dump
    )


def passes(metrics: dict, thresholds: Optional[Thresholds] = None) -> Tuple[bool, List[str]]:
    """Check metrics against thresholds; return (ok, list_of_failures)."""
    t = thresholds or Thresholds()
    o = metrics["overall"]
    failures: List[str] = []
    if o["identity_accuracy"] < t.min_identity_accuracy:
        failures.append(f"identity_accuracy {o['identity_accuracy']:.4f} < {t.min_identity_accuracy}")
    if o["idf1"] < t.min_idf1:
        failures.append(f"idf1 {o['idf1']:.4f} < {t.min_idf1}")
    if o["total_id_switches"] > t.max_id_switches:
        failures.append(f"id_switches {o['total_id_switches']} > {t.max_id_switches}")
    if o["min_coverage"] < t.min_coverage:
        failures.append(f"min_coverage {o['min_coverage']:.4f} < {t.min_coverage}")
    if not np.isnan(o["worst_angle_mae"]) and o["worst_angle_mae"] > t.max_angle_mae:
        failures.append(f"worst_angle_mae {o['worst_angle_mae']:.3f} > {t.max_angle_mae}")
    if o["extra_id_frames"] > t.max_extra_id_frames:
        failures.append(f"extra_id_frames {o['extra_id_frames']} > {t.max_extra_id_frames}")
    return (len(failures) == 0, failures)


def format_report(metrics: dict) -> str:
    """Human-readable multi-line report."""
    o = metrics["overall"]
    lines = ["=" * 64, "WHISKER LINKING EVALUATION", "=" * 64]
    lines.append(f"Identity accuracy : {o['identity_accuracy']:.4f}")
    lines.append(f"MOTA              : {o['mota']:.4f}")
    lines.append(f"IDF1              : {o['idf1']:.4f}")
    lines.append(f"ID switches       : {o['total_id_switches']}")
    lines.append(f"GT detections     : {o['total_gt_detections']}  "
                 f"(miss={o['miss']} fp={o['false_positive']} mismatch={o['mismatch']})")
    lines.append(f"Angle MAE (deg)   : mean={o['mean_angle_mae']:.3f} worst={o['worst_angle_mae']:.3f}")
    lines.append(f"Min coverage      : {o['min_coverage']:.4f}")
    lines.append(f"Angle jumps       : pred={o['pred_angle_jumps']} gt={o['gt_angle_jumps']}")
    lines.append(f"Spurious ids      : {o['extra_ids']} ({o['extra_id_frames']} frames, {o['extra_id_rows']} rows)")
    lines.append(f"wid map (pred->gt): {metrics['wid_map']}")
    lines.append("-" * 64)
    lines.append("Per ground-truth whisker:")
    for gw, v in sorted(metrics["per_gt_whisker"].items(), key=lambda kv: int(kv[0])):
        mae = "nan" if np.isnan(v["angle_mae"]) else f"{v['angle_mae']:.2f}"
        lines.append(f"  gt {gw:>3} <- pred {v['mapped_pred_wid']:>3} | "
                     f"cov={v['coverage']:.3f} idsw={v['id_switches']} "
                     f"frag={v['fragmentation']} mae={mae}")
    return "\n".join(lines)


def plot_diff(metrics: dict, out_path: str) -> None:
    """Save a fid x gt_whisker raster: green = identity-correct, red = wrong/miss.

    Localises exactly where linking breaks (e.g. the crossing region).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mt = metrics["_match_table"]
    gt = mt[mt["gt_wid"] != NO_GT].copy()
    wid_map = metrics["wid_map"]
    gt["pred_gt"] = gt["pred_wid"].map(wid_map)
    gt["state"] = np.where(~gt["matched"], 0,                       # miss
                  np.where(gt["pred_gt"] == gt["gt_wid"], 2, 1))    # 2=ok 1=wrong

    gt_wids = sorted(gt["gt_wid"].unique())
    fids = np.arange(int(gt["fid"].min()), int(gt["fid"].max()) + 1)
    grid = np.full((len(gt_wids), len(fids)), -1.0)
    fpos = {f: i for i, f in enumerate(fids)}
    wpos = {w: i for i, w in enumerate(gt_wids)}
    for _, r in gt.iterrows():
        grid[wpos[int(r["gt_wid"])], fpos[int(r["fid"])]] = r["state"]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#dddddd", "#c0392b", "#e67e22", "#27ae60"])  # -1,0,1,2
    fig, ax = plt.subplots(figsize=(min(20, max(8, len(fids) / 40)), 1 + 0.6 * len(gt_wids)))
    ax.imshow(grid + 1, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3,
              extent=[fids[0], fids[-1], len(gt_wids) - 0.5, -0.5])
    ax.set_yticks(range(len(gt_wids)))
    ax.set_yticklabels([f"gt {w}" for w in gt_wids])
    ax.set_xlabel("frame (fid)")
    ax.set_title("Linking correctness  (green=ok, orange=wrong id, red=miss, grey=absent)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate(pred_path: str, gt_path: str, *, gate_px: float = 25.0,
             angle_jump_thresh: float = 20.0, wid_col: str = "wid") -> dict:
    """Load two parquet files and compute metrics."""
    pred_df = pd.read_parquet(pred_path)
    gt_df = pd.read_parquet(gt_path)
    return compute_metrics(pred_df, gt_df, gate_px=gate_px,
                           angle_jump_thresh=angle_jump_thresh, wid_col=wid_col)


def _strip_internal(metrics: dict) -> dict:
    """Return a json-serialisable copy without the retained match table."""
    out = {k: v for k, v in metrics.items() if not k.startswith("_")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate whisker linking quality.")
    ap.add_argument("--pred", required=True, help="Predicted tracking parquet")
    ap.add_argument("--gt", required=True, help="Ground-truth (corrected) parquet")
    ap.add_argument("--gate-px", type=float, default=25.0, help="Follicle match gate (px)")
    ap.add_argument("--angle-jump-thresh", type=float, default=20.0, help="deg/frame jump threshold")
    ap.add_argument("--wid-col", default="wid", help="Identity column (default wid)")
    ap.add_argument("--diff-plot", help="Path to save the correctness raster PNG")
    ap.add_argument("--json", dest="json_out", help="Path to save metrics as JSON")
    args = ap.parse_args()

    metrics = evaluate(args.pred, args.gt, gate_px=args.gate_px,
                       angle_jump_thresh=args.angle_jump_thresh, wid_col=args.wid_col)
    print(format_report(metrics))
    if args.diff_plot:
        plot_diff(metrics, args.diff_plot)
        print(f"\nDiff raster saved to {args.diff_plot}")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(_strip_internal(metrics), f, indent=2)
        print(f"Metrics JSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
