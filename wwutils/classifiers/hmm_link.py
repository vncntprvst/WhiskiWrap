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


def link_whiskers_hmm(combined_parquet: str, wt_dir: str, base_name: str,
                      side_faces: Dict[str, str], whiskerpad=None,
                      n_per_side: Optional[Dict[str, int]] = None,
                      output_path: Optional[str] = None, **classify_kw) -> Optional[str]:
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

    out = apply_hmm_identity(combined, hmm).sort_values(["fid", "wid"])
    output_path = output_path or combined_parquet.replace(".parquet", "_updated.parquet")
    out.to_parquet(output_path)
    print(f"[hmm_link] saved {len(out)} rows to {output_path}")
    return output_path
