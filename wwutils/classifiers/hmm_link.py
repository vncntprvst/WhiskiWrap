"""Cross-chunk whisker identity using whisk's per-chunk classify + HMM reclassify.

whisk's ``classify -n N`` + ``reclassify`` assign a temporally-consistent identity
*within* each traced chunk (the HMM models identity across the chunk's frames).
Empirically this is far more stable than re-deriving identity from scratch, but the
per-chunk identity numbering is independent across chunks. This module:

1. estimates the number of whiskers ``N`` per face side from combined tracking data,
2. (re)runs ``classify -n N`` + ``reclassify`` on each chunk's ``.measurements``,
3. stitches per-chunk identities into a single global identity across chunks
   (a small assignment on boundary follicle position), and
4. joins the resulting global identity back onto a combined parquet by geometry.

Requires the whisk binaries (from the ``whisk-janelia`` package) and WhiskiWrap's
``mfile_io`` for reading ``.measurements`` files.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Whisk binary helpers
# --------------------------------------------------------------------------- #
def _bin(name: str) -> str:
    """Full path to a whisk executable (with .exe on Windows)."""
    import whisk
    exe = name + (".exe" if os.name == "nt" else "")
    return str(whisk.get_whisk_bin_path() / exe)


def reclassify_measurements(meas_path: str, face: str, n: int, *,
                            px2mm: float = 0.06, limit: str = "2.0:40.0",
                            follicle: Optional[str] = "150") -> None:
    """Run ``classify -n N`` then ``reclassify`` (HMM) on one ``.measurements`` file.

    The file is modified in place (whisk writes identity into it).
    """
    classify = [_bin("classify"), meas_path, meas_path, face, "--px2mm", str(px2mm), "-n", str(n)]
    if limit:
        classify.append(f"--limit{limit}")
    if follicle is not None:
        classify += ["--follicle", str(follicle)]
    reclassify = [_bin("reclassify"), meas_path, meas_path, "-n", str(n)]
    cwd = os.path.dirname(meas_path) or "."
    subprocess.run(classify, cwd=cwd, capture_output=True)
    subprocess.run(reclassify, cwd=cwd, capture_output=True)


def read_measurements(meas_path: str) -> pd.DataFrame:
    """Read a ``.measurements`` file into a DataFrame.

    Columns (whisk measurements layout): state (identity), fid (frame), wid,
    length, score, angle, curvature, follicle_x/y, tip_x/y.
    """
    from WhiskiWrap.mfile_io import MeasurementsTable
    a = MeasurementsTable(meas_path).asarray()
    return pd.DataFrame({
        "state": a[:, 0].astype(int), "fid": a[:, 1].astype(int), "wid": a[:, 2].astype(int),
        "length": a[:, 3], "score": a[:, 4], "angle": a[:, 5], "curvature": a[:, 6],
        "follicle_x": a[:, 7], "follicle_y": a[:, 8], "tip_x": a[:, 9], "tip_y": a[:, 10],
    })


# --------------------------------------------------------------------------- #
# N estimation
# --------------------------------------------------------------------------- #
def estimate_n_per_side(df: pd.DataFrame, *, length_frac: float = 0.5,
                        min_frame_ratio: float = 0.5) -> Dict[str, int]:
    """Estimate the number of whiskers per face side.

    A whisker is "real" if, among detections longer than ``length_frac`` x the
    longest median length, it appears in at least ``min_frame_ratio`` of frames.
    Falls back to the median per-frame count of long detections.
    """
    out: Dict[str, int] = {}
    for side, g in df.groupby("face_side"):
        thr = g.groupby("wid")["length"].median().max() * length_frac
        gl = g[g["length"] > thr]
        if gl.empty:
            out[side] = 1
            continue
        nf = gl["fid"].nunique()
        counts = gl.groupby("wid")["fid"].nunique()
        n = int((counts >= min_frame_ratio * nf).sum())
        if n == 0:
            n = int(round(gl.groupby("fid").size().median()))
        out[side] = max(n, 1)
    return out


# --------------------------------------------------------------------------- #
# Cross-chunk stitching
# --------------------------------------------------------------------------- #
def stitch_chunk_identities(chunks: List[Tuple[int, pd.DataFrame]], *,
                            axis: str = "follicle_y") -> pd.DataFrame:
    """Assign a global identity (``gid``) across chunks.

    ``chunks`` is a list of ``(chunk_start, df)`` ordered by chunk_start, where
    each df contains only identified detections (``state`` >= 0) with a global
    ``fid`` and the ``axis`` column. Identities are matched chunk-to-chunk by mean
    position along ``axis`` (a small Hungarian assignment per boundary).
    """
    chunks = sorted(chunks, key=lambda t: t[0])
    out = []
    ref: Optional[Dict[int, float]] = None  # global_id -> last axis position
    for _, d in chunks:
        if d.empty:
            continue
        means = d.groupby("state")[axis].mean()
        if ref is None:
            mapping = {st: i for i, st in enumerate(means.sort_values().index)}
        else:
            rids = list(ref.keys()); rpos = np.array([ref[r] for r in rids])
            sids = list(means.index); spos = means.values
            cost = np.abs(rpos[:, None] - spos[None, :])
            ri, ci = linear_sum_assignment(cost)
            mapping = {sids[c]: rids[r] for r, c in zip(ri, ci)}
            # any unmatched current identities get new global ids
            nxt = (max(ref.keys()) + 1) if ref else 0
            for st in means.index:
                if st not in mapping:
                    mapping[st] = nxt; nxt += 1
        dd = d.copy(); dd["gid"] = dd["state"].map(mapping)
        out.append(dd)
        ref = {mapping[st]: means.loc[st] for st in means.index}
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def relabel_side_with_hmm(wt_dir: str, base_name: str, side: str, face: str, n: int,
                          coord_offset: Tuple[float, float] = (0.0, 0.0),
                          **classify_kw) -> pd.DataFrame:
    """classify+reclassify every chunk of one side and stitch to a global identity.

    ``coord_offset`` (x, y) is the per-side crop offset (whiskerpad
    ImageCoordinates): the ``.measurements`` follicle coordinates are in cropped
    per-side space, so the offset is added to convert them to full-frame space to
    match a combined parquet.

    Returns a DataFrame of identified detections with columns:
    face_side, fid (global), wid, follicle_x, follicle_y, angle, length, gid.
    """
    ox, oy = coord_offset
    pattern = os.path.join(wt_dir, f"{base_name}_{side}_*.measurements")
    chunks = []
    for meas in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{8})\.measurements$", os.path.basename(meas))
        if not m:
            continue
        chunk_start = int(m.group(1))
        reclassify_measurements(meas, face, n, **classify_kw)
        d = read_measurements(meas)
        d = d[d["state"] >= 0].copy()
        d["fid"] = d["fid"] + chunk_start
        d["follicle_x"] = d["follicle_x"] + ox
        d["follicle_y"] = d["follicle_y"] + oy
        chunks.append((chunk_start, d))
    if not chunks:
        return pd.DataFrame()
    stitched = stitch_chunk_identities(chunks)
    stitched["face_side"] = side
    return stitched


def apply_hmm_identity(combined: pd.DataFrame, hmm: pd.DataFrame, *,
                       gate_px: float = 20.0) -> pd.DataFrame:
    """Join HMM global identity (``gid``) onto ``combined`` rows by follicle geometry.

    For each (fid, face_side) the HMM detections are matched to the combined-parquet
    detections by nearest follicle (within ``gate_px``). Matched rows get the HMM
    identity in ``wid`` (raw id preserved in ``label``); unmatched combined rows are
    dropped (they were not assigned an identity by whisk).
    """
    keep = []
    for (fid, side), c in combined.groupby(["fid", "face_side"]):
        h = hmm[(hmm["fid"] == fid) & (hmm["face_side"] == side)]
        if h.empty:
            continue
        cx = c[["follicle_x", "follicle_y"]].to_numpy(float)
        hx = h[["follicle_x", "follicle_y"]].to_numpy(float)
        dist = np.sqrt(((cx[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
        ci, hi = linear_sum_assignment(dist)
        for r, k in zip(ci, hi):
            if dist[r, k] <= gate_px:
                row = c.iloc[r].copy()
                row["label"] = row["wid"]
                row["wid"] = int(h.iloc[k]["gid"])
                keep.append(row)
    return pd.DataFrame(keep)


def _side_offsets(whiskerpad) -> Dict[str, Tuple[float, float]]:
    """Per-side crop offsets (x, y) from a whiskerpad JSON path or dict."""
    import json
    if isinstance(whiskerpad, str) and os.path.isfile(whiskerpad):
        with open(whiskerpad) as f:
            whiskerpad = json.load(f)
    offsets: Dict[str, Tuple[float, float]] = {}
    if isinstance(whiskerpad, dict):
        for pad in whiskerpad.get("whiskerpads", []):
            ic = pad.get("ImageCoordinates", [0, 0, 0, 0])
            offsets[pad["FaceSide"].lower()] = (ic[0], ic[1])
    return offsets


def filter_follicle_outliers(df: pd.DataFrame, max_dist: float = 40.0) -> pd.DataFrame:
    """Drop detections whose follicle is far from their identity's base.

    whisk's ``classify -n N`` is forced to output N identities every frame, so when
    a real whisker is occluded (e.g. by a cue-tip) it labels noise (a tip fragment,
    a stray hair, a cotton strand) with the freed-up identity. Such noise has a
    follicle far from that identity's true base, so we reject detections more than
    ``max_dist`` px from the identity's robust (median) follicle position. This
    leaves a gap for the occluded frames rather than a wrong label.
    """
    if df.empty:
        return df
    keep = []
    for wid, g in df.groupby("wid"):
        cx, cy = g["follicle_x"].median(), g["follicle_y"].median()
        d = np.hypot(g["follicle_x"] - cx, g["follicle_y"] - cy)
        keep.append(g[d <= max_dist])
    return pd.concat(keep) if keep else df


def filter_length_outliers(df: pd.DataFrame, min_frac: float = 0.4) -> pd.DataFrame:
    """Drop detections much shorter than their identity's typical length.

    When a real whisker is occluded, ``classify`` may label a short stray hair
    sitting near its base with that identity (so the follicle filter misses it).
    A real whisker's length is fairly stable, so detections shorter than
    ``min_frac`` x the identity's median length are rejected.
    """
    if df.empty:
        return df
    keep = []
    for wid, g in df.groupby("wid"):
        thr = g["length"].median() * min_frac
        keep.append(g[g["length"] >= thr])
    return pd.concat(keep) if keep else df


def estimate_follicle_gate(df: pd.DataFrame, frac: float = 0.25) -> float:
    """Empirical follicle gate (px) ~ a fraction of the median whisker length.

    Whisker length, inter-whisker spacing and follicle motion all scale with the
    camera/lens/zoom, so anchoring the gate to the median traced length makes it
    robust across setups (no hard-coded pixel value). Real follicle motion is a
    small fraction of length; noise (cotton/tips) sits ~a whisker length away.
    """
    # frac is a fraction of the median traced length; the gate is the per-frame
    # follicle continuity radius (the base barely moves frame-to-frame, so a small
    # fraction suffices and rejects noise that sits farther from the track).
    med_len = float(df["length"].median()) if len(df) else 0.0
    return max(frac * med_len, 1.0)


def reassign_by_tracking(df: pd.DataFrame, gate_px: float, max_missed: int = 30) -> pd.DataFrame:
    """Forward temporal re-assignment of identities (continuity-aware).

    Per side, processes frames in order and assigns each detection to the closest
    identity *predicted position* (its last seen follicle, or its global centroid if
    not seen within ``max_missed`` frames). Temporal continuity keeps whiskers on
    their own track even when their bases are close (~px apart) or one is occluded,
    fixing the swaps/shifts that independent per-frame matching produces. Detections
    farther than ``gate_px`` from every prediction are dropped (noise/cotton).
    """
    if df.empty:
        return df
    cen = df.groupby("wid")[["follicle_x", "follicle_y"]].median()
    id_side = df.groupby("wid")["face_side"].first()
    rows = []
    for side in df["face_side"].unique():
        ids = [w for w in cen.index if id_side[w] == side]
        if not ids:
            continue
        last = {w: cen.loc[w].to_numpy(float) for w in ids}
        missed = {w: 0 for w in ids}
        sd = df[df["face_side"] == side]
        for fid in sorted(sd["fid"].unique()):
            g = sd[sd["fid"] == fid]
            G = g[["follicle_x", "follicle_y"]].to_numpy(float)
            P = np.array([last[w] if missed[w] <= max_missed else cen.loc[w].to_numpy(float)
                          for w in ids])
            D = np.sqrt(((G[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            gi, ci = linear_sum_assignment(D)
            matched = set()
            for r, k in zip(gi, ci):
                if D[r, k] <= gate_px:
                    w = ids[k]
                    row = g.iloc[r].copy(); row["wid"] = int(w); rows.append(row)
                    last[w] = G[r]; missed[w] = 0; matched.add(k)
            for j, w in enumerate(ids):
                if j not in matched:
                    missed[w] += 1
    return pd.DataFrame(rows)


def reassign_by_centroid(df: pd.DataFrame, gate_px: float) -> pd.DataFrame:
    """Per frame, assign each detection to the nearest identity centroid (Hungarian).

    Corrects within-frame identity swaps (e.g. whisker 1 mislabelled 2 when whisker
    2 is occluded) using each identity's robust (median) follicle centroid, and
    drops detections farther than ``gate_px`` from every centroid (noise). One-to-one
    per (frame, side); occluded identities simply go unmatched (a clean gap).
    """
    if df.empty:
        return df
    cen = df.groupby("wid")[["follicle_x", "follicle_y"]].median()
    id_side = df.groupby("wid")["face_side"].first()
    rows = []
    for (fid, side), g in df.groupby(["fid", "face_side"]):
        sid_ids = [w for w in cen.index if id_side[w] == side]
        if not sid_ids:
            continue
        C = cen.loc[sid_ids].to_numpy(float)
        G = g[["follicle_x", "follicle_y"]].to_numpy(float)
        D = np.sqrt(((G[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        gi, ci = linear_sum_assignment(D)
        for r, k in zip(gi, ci):
            if D[r, k] <= gate_px:
                row = g.iloc[r].copy()
                row["wid"] = int(sid_ids[k])
                rows.append(row)
    return pd.DataFrame(rows)


def _runs(sorted_vals: List[int]) -> List[List[int]]:
    """Split a sorted list of ints into maximal consecutive runs."""
    runs: List[List[int]] = []
    for v in sorted_vals:
        if runs and v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    return runs


def bridge_gaps(out: pd.DataFrame, combined: pd.DataFrame, *, max_gap: int = 20,
                gate_px: float = 25.0, min_length_frac: float = 0.4) -> pd.DataFrame:
    """Recover identities for frames where a *visible* whisker was left unclassified.

    For each identity, short gaps (<= ``max_gap`` frames) between present frames are
    filled by searching the combined detections for an as-yet-unassigned, long
    enough whisker near the interpolated follicle position. This recovers whiskers
    that whisk failed to classify (``state = -1``) without inventing a whisker for
    genuinely occluded frames (no suitable detection exists there).
    """
    if out.empty:
        return out
    assigned = set(out.index)
    new_rows = []
    for side in out["face_side"].unique():
        cs = combined[combined["face_side"] == side]
        for wid, g in out[out["face_side"] == side].groupby("wid"):
            g = g.sort_values("fid")
            present = set(g["fid"].tolist())
            med_len = g["length"].median()
            fol = g.drop_duplicates("fid").set_index("fid")[["follicle_x", "follicle_y"]]
            fmin, fmax = g["fid"].iloc[0], g["fid"].iloc[-1]
            missing = [f for f in range(int(fmin), int(fmax) + 1) if f not in present]
            for run in _runs(missing):
                if len(run) > max_gap:
                    continue
                a, b = run[0] - 1, run[-1] + 1
                fa, fb = fol.loc[a], fol.loc[b]
                for f in run:
                    t = (f - a) / (b - a)
                    ex = fa["follicle_x"] * (1 - t) + fb["follicle_x"] * t
                    ey = fa["follicle_y"] * (1 - t) + fb["follicle_y"] * t
                    cand = cs[(cs["fid"] == f) & (~cs.index.isin(assigned))
                              & (cs["length"] >= min_length_frac * med_len)]
                    if cand.empty:
                        continue
                    d = np.hypot(cand["follicle_x"] - ex, cand["follicle_y"] - ey)
                    if d.min() <= gate_px:
                        idx = d.idxmin()
                        row = cs.loc[idx].copy()
                        row["label"] = row["wid"]
                        row["wid"] = int(wid)
                        new_rows.append(row)
                        assigned.add(idx)
    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)])
    return out


def link_whiskers_hmm(combined_parquet: str, wt_dir: str, base_name: str,
                      side_faces: Dict[str, str], whiskerpad=None,
                      n_per_side: Optional[Dict[str, int]] = None,
                      output_path: Optional[str] = None,
                      follicle_gate_frac: float = 0.15, follicle_max_dist: Optional[float] = None,
                      length_min_frac: float = 0.4, bridge_max_gap: int = 20,
                      **classify_kw) -> Optional[str]:
    """End-to-end HMM linking: estimate N, reclassify chunks, stitch, join, save.

    ``side_faces`` maps face side -> the ``--face`` argument used at trace time
    (e.g. {"left": "left", "right": "right"}). ``whiskerpad`` (JSON path or dict)
    supplies per-side crop offsets so cropped measurement coordinates align with
    the full-frame combined parquet. Writes ``*_updated.parquet`` (or
    ``output_path``) and returns its path.
    """
    combined = pd.read_parquet(combined_parquet)
    if n_per_side is None:
        n_per_side = estimate_n_per_side(combined)
    offsets = _side_offsets(whiskerpad) if whiskerpad is not None else {}
    print(f"[hmm_link] whiskers per side: {n_per_side}")

    hmm_parts = []
    base = 0
    for side in sorted(side_faces):
        n = n_per_side.get(side, 1)
        part = relabel_side_with_hmm(wt_dir, base_name, side, side_faces[side], n,
                                     coord_offset=offsets.get(side, (0.0, 0.0)), **classify_kw)
        if part.empty:
            print(f"[hmm_link] {side}: no identities produced.")
            continue
        part["gid"] = part["gid"] + base                 # keep sides disjoint
        base += part["gid"].nunique()
        hmm_parts.append(part)
        print(f"[hmm_link] {side}: {part['gid'].nunique()} whiskers, {len(part)} detections.")
    if not hmm_parts:
        return None
    hmm = pd.concat(hmm_parts, ignore_index=True)

    out = apply_hmm_identity(combined, hmm)
    if length_min_frac:
        before = len(out)
        out = filter_length_outliers(out, length_min_frac)
        print(f"[hmm_link] length-outlier filter dropped {before - len(out)} detections.")
    # Empirical follicle gate (scales with camera/zoom via whisker length).
    gate = follicle_max_dist if follicle_max_dist else estimate_follicle_gate(out, follicle_gate_frac)
    print(f"[hmm_link] follicle gate = {gate:.1f} px")
    before = len(out)
    out = reassign_by_tracking(out, gate)   # temporal continuity: fix swaps/shifts + drop far noise
    print(f"[hmm_link] tracking re-assignment dropped {before - len(out)} far detections.")
    if bridge_max_gap:
        before = len(out)
        out = bridge_gaps(out, combined, max_gap=bridge_max_gap, gate_px=gate,
                          min_length_frac=length_min_frac or 0.4)
        print(f"[hmm_link] gap-bridging recovered {len(out) - before} detections.")
    out = out.sort_values(["fid", "wid"])
    output_path = output_path or combined_parquet.replace(".parquet", "_updated.parquet")
    out.to_parquet(output_path)
    print(f"[hmm_link] saved {len(out)} rows to {output_path}")
    return output_path
