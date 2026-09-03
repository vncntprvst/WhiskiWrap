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
                        min_frame_ratio: float = 0.5,
                        rescue_frame_ratio: float = 2.0,   # >1 == off; see below
                        rescue_peak_frac: float = 0.70,
                        rescue_peak_q: float = 0.95) -> Dict[str, int]:
    """Estimate the number of whiskers per face side.

    A whisker is "real" if, among detections longer than ``length_frac`` x the
    longest median length, it appears in at least ``min_frame_ratio`` of frames.
    Falls back to the median per-frame count of long detections.

    RESCUE PASS -- OFF BY DEFAULT, and it should stay off until the premise below
    is fixed. Set ``rescue_frame_ratio`` <= 1 to enable it.

    The motivating problem is real: on a poke clip the anterior whisker on the
    right reached 153 px but had a median of 60 px against a 95 px threshold, so
    the side was scored 2 while three whiskers were visible. Rescuing labels whose
    upper-quantile length is long does fix that case.

    But it over-counts, because the premise is wrong. ``wid`` here is a per-chunk
    label, not a persistent identity: across frames one label covers a full shaft
    in some and a 36 px fur stub in others. On a baseline clip five labels persist
    across a side that has only three long whiskers, and the rescue promoted a
    fourth. Counting labels is not counting whiskers, so no threshold on a
    per-label length statistic can be trusted -- the animal has three long
    whiskers per side and the estimate must land on three.

    Use ``n_per_side`` (whisker_tracking.py --n-whiskers) when the count is known.
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
        kept = set(counts[counts >= min_frame_ratio * nf].index)

        # --- rescue pass: persistent labels that peak long enough to be a whisker
        rescued = []
        if rescue_frame_ratio <= 1.0:
            med = g.groupby("wid")["length"].median()
            peak = g.groupby("wid")["length"].quantile(rescue_peak_q)
            nf_all = g["fid"].nunique()
            allcnt = g.groupby("wid")["fid"].nunique()
            peak_thr = peak.max() * rescue_peak_frac
            for w in med.index:
                if w in kept:
                    continue
                if peak.get(w, 0) > peak_thr and allcnt.get(w, 0) >= rescue_frame_ratio * nf_all:
                    kept.add(w)
                    rescued.append(w)

        n = len(kept)
        if n == 0:
            n = int(round(gl.groupby("fid").size().median()))
        out[side] = max(n, 1)
        if rescued:
            pk = g.groupby("wid")["length"].quantile(rescue_peak_q)
            mm = g.groupby("wid")["length"].median()
            desc = ", ".join(f"wid {w} (median {mm[w]:.0f}px, peak {pk[w]:.0f}px)"
                             for w in rescued)
            print(f"[hmm_link] {side}: rescued {len(rescued)} persistent long-peaking "
                  f"label(s): {desc}")

        # Report near-misses. The median-length test rejects a real whisker that is
        # only intermittently traced in full: on a poke clip the anterior right
        # whisker had a median of 60 px against a 95 px threshold while reaching
        # 150 px in the frames where it was traced whole, so the side was linked
        # with 2 identities while 3 whiskers were visible. Nothing here can tell
        # such a whisker from a persistent fur artefact, so say what was discarded
        # and let the caller override with n_per_side rather than fail silently.
        gall = g[g["length"] > thr * 0.4]
        if not gall.empty:
            allcnt = gall.groupby("wid")["fid"].nunique()
            persistent = allcnt[allcnt >= min_frame_ratio * df[df["face_side"] == side]["fid"].nunique()]
            near = [w for w in persistent.index
                    if w not in counts.index or counts.get(w, 0) < min_frame_ratio * nf]
            if near:
                stats = g[g["wid"].isin(near)].groupby("wid")["length"]
                desc = ", ".join(
                    f"wid {w} (median {stats.median()[w]:.0f}px, max {stats.max()[w]:.0f}px)"
                    for w in sorted(near, key=lambda w: -stats.max()[w])[:5])
                print(f"[hmm_link] {side}: n={out[side]}; rejected but persistent: {desc}"
                      f"  -- pass n_per_side to override")
    return out


# --------------------------------------------------------------------------- #
# Cross-chunk stitching
# --------------------------------------------------------------------------- #
def _boundary_pos(d: pd.DataFrame, which: str, k: int) -> pd.DataFrame:
    """Per-state median 2D follicle over the first/last ``k`` frames of a chunk.

    Falls back to the whole-chunk mean for a state absent from the boundary window.
    """
    fids = sorted(d["fid"].unique())
    keep = set(fids[-k:] if which == "end" else fids[:k])
    m = d[d["fid"].isin(keep)].groupby("state")[["follicle_x", "follicle_y"]].median()
    full = d.groupby("state")[["follicle_x", "follicle_y"]].mean()
    for st in full.index:
        if st not in m.index:
            m.loc[st] = full.loc[st]
    return m


def _match_states(ref: Dict[int, np.ndarray], pos: pd.DataFrame,
                  margin: float) -> Tuple[Dict[int, int], bool]:
    """Hungarian-match a chunk's per-state positions to reference gid positions.

    Returns (state->gid mapping, confident) where ``confident`` is True only if
    every state's nearest reference is clearly closer than its second nearest (by
    ``margin`` px) -- i.e. the match is unambiguous.
    """
    rids = list(ref.keys()); rmat = np.array([ref[r] for r in rids])
    sids = list(pos.index)
    smat = pos.loc[sids, ["follicle_x", "follicle_y"]].to_numpy(float)
    cost = np.linalg.norm(rmat[:, None, :] - smat[None, :, :], axis=2)  # [nref, nsid]
    ri, ci = linear_sum_assignment(cost)
    mapping = {sids[c]: rids[r] for r, c in zip(ri, ci)}
    confident = True
    for c in range(len(sids)):
        col = np.sort(cost[:, c])
        if len(col) > 1 and col[1] - col[0] < margin:
            confident = False
            break
    return mapping, confident


def stitch_chunk_identities(chunks: List[Tuple[int, pd.DataFrame]], *,
                            axis: str = "follicle_y", boundary_frames: int = 25,
                            confident_margin: float = 10.0) -> pd.DataFrame:
    """Assign a global identity (``gid``) across chunks.

    ``chunks`` is a list of ``(chunk_start, df)`` ordered by chunk_start, where each
    df contains only identified detections (``state`` >= 0) with a global ``fid``.
    Across each seam, identities are matched by 2D follicle position using a hybrid
    of two signatures:
      * **boundary** -- median over the last ``boundary_frames`` of one chunk vs the
        first of the next; this follows the actual transition and disambiguates close
        whiskers (e.g. id1/id2 ~15 px apart) that a whole-chunk mean flips when the
        per-chunk means happen to cross.
      * **whole-chunk mean** -- robust when a whisker is occluded near the boundary
        (the boundary frames are then unreliable).
    The boundary match is used when it is unambiguous (margin > ``confident_margin``);
    otherwise it falls back to the whole-chunk-mean match.
    """
    chunks = sorted(chunks, key=lambda t: t[0])
    out = []
    ref_b: Optional[Dict[int, np.ndarray]] = None  # gid -> boundary pos at prev end
    ref_m: Optional[Dict[int, np.ndarray]] = None  # gid -> whole-chunk mean pos
    for _, d in chunks:
        if d.empty:
            continue
        start_b = _boundary_pos(d, "start", boundary_frames)
        mean_pos = d.groupby("state")[["follicle_x", "follicle_y"]].mean()
        if ref_b is None:
            mapping = {st: i for i, st in enumerate(start_b[axis].sort_values().index)}
        else:
            mapping, confident = _match_states(ref_b, start_b, confident_margin)
            if not confident:
                mapping, _ = _match_states(ref_m, mean_pos, confident_margin)
        # any state not matched (new/unmatched identity) gets a fresh global id
        nxt = (max(mapping.values()) + 1) if mapping else 0
        if ref_b:
            nxt = max(nxt, max(ref_b.keys()) + 1)
        for st in d["state"].unique():
            if st not in mapping:
                mapping[st] = nxt; nxt += 1
        dd = d.copy(); dd["gid"] = dd["state"].map(mapping)
        out.append(dd)
        end_b = _boundary_pos(d, "end", boundary_frames)
        ref_b = {mapping[st]: end_b.loc[st, ["follicle_x", "follicle_y"]].to_numpy(float)
                 for st in end_b.index}
        ref_m = {mapping[st]: mean_pos.loc[st, ["follicle_x", "follicle_y"]].to_numpy(float)
                 for st in mean_pos.index}
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


def filter_follicle_outliers(df: pd.DataFrame, max_dist: float = 40.0,
                             follicle_window: int = 31) -> pd.DataFrame:
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
        g = g.sort_values("fid")
        # Local (rolling) median follicle so the reference tracks the follicle's own
        # motion. The head is fixed, but the whiskerpad -- and thus the follicle
        # positions -- shifts during whisking, so a global whole-clip median wrongly
        # drops real whiskers in extreme-phase frames whose base sits at the edge of
        # its range. Detections far from the *local* base are still rejected (cotton /
        # stray hairs).
        cx = g["follicle_x"].rolling(follicle_window, center=True, min_periods=1).median()
        cy = g["follicle_y"].rolling(follicle_window, center=True, min_periods=1).median()
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


def filter_angle_outliers(df: pd.DataFrame, window: int = 31, angle_k: float = 4.0,
                          len_lo: float = 0.6, len_hi: float = 1.7) -> pd.DataFrame:
    """Drop brief detections whose angle AND length both depart from the local trend.

    A wrong detection that momentarily grabs an identity (e.g. a stray nearly
    orthogonal to the whisker) shows up as a single frame whose angle is far from
    the identity's local rolling-median angle *and* whose length is abnormal.
    Requiring BOTH anomalies spares real fast-whisking frames (angle swings a lot
    but length stays normal). Compared per identity against a centered rolling
    median so the whisking sweep itself is not flagged.
    """
    if df.empty:
        return df

    def adiff(a, b):
        return np.abs(((a - b + 180) % 360) - 180)

    keep = []
    for wid, g in df.groupby("wid"):
        g = g.sort_values("fid")
        rma = g["angle"].rolling(window, center=True, min_periods=5).median()
        rml = g["length"].rolling(window, center=True, min_periods=5).median()
        amad = adiff(g["angle"], rma)
        scale = max(np.nanmedian(amad), 3.0)
        ang_out = amad > angle_k * scale
        len_out = (g["length"] < len_lo * rml) | (g["length"] > len_hi * rml)
        keep.append(g[~(ang_out & len_out)])
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


def _n_from_classify_labels(combined, min_frame_frac: float = 0.5) -> Dict[str, int]:
    """Whiskers per side, taken from the identities classify assigned.

    classify labels each segment -1 (not a whisker) or 0,1,2... (whisker n), and
    the number of distinct non-negative labels on a side IS its whisker count --
    already decided using the length and follicle tests it was configured with.
    Deriving it again from length statistics throws that away and does worse.

    classify numbers its whiskers 0..N-1, so N = max(label) + 1 is its answer --
    not the number of labels that happen to be common. A whisker that is occluded
    for part of the clip (by the cuetip, say) still exists; requiring each identity
    in >=50% of frames dropped exactly that one and returned 2 for a side with 3.

    ``min_frame_frac`` is therefore only used to report thin identities, not to
    exclude them. Returns {} when there is no usable label column, so the caller
    can fall back to the older estimate rather than guessing 1.
    """
    if "label" not in combined.columns or "face_side" not in combined.columns:
        return {}
    out: Dict[str, int] = {}
    for side, g in combined.groupby("face_side"):
        w = g[g["label"] >= 0]
        if w.empty:
            continue
        out[side] = int(w["label"].max()) + 1
        nf = g["fid"].nunique()
        per_label = w.groupby("label")["fid"].nunique()
        thin = {int(k): f"{100.0 * v / nf:.0f}%" for k, v in per_label.items()
                if v < min_frame_frac * nf}
        if thin:
            print(f"[hmm_link] {side}: identity present in few frames "
                  f"(kept anyway -- occlusion is not absence): {thin}")
    return out


def hmm_backbone(combined_parquet, wt_dir, base_name, side_faces, *,
                 whiskerpad=None, n_per_side=None, **classify_kw):
    """Run the whisk-HMM backbone (classify+reclassify per chunk, stitch, join) and
    return ``(combined_df, out_df)`` where ``out`` is post-``apply_hmm_identity`` (the raw
    candidate labeling, before any coverage filtering / identity re-rank). Returns None if
    no identities were produced. Exposed so the autotune loop can cache this expensive step
    once and apply learned add-ons offline."""
    combined = pd.read_parquet(combined_parquet)

    # Honour whisk's own classification. `label` carries what classify/reclassify
    # decided: -1 for "traced, but not a whisker" (fur, stubs, noise), 0,1,2... for
    # whiskers. classify is good at this -- on a poke clip it marked 70-92% of
    # segments -1 and left exactly the right whiskers, median length 263 px against
    # 44 px for what it rejected.
    #
    # Linking without this filter means reconstructing identity from ~30
    # undifferentiated segments per frame when classify had already narrowed it to
    # three, which is where the spurious extra identities and the unstable whisker
    # count came from.
    if "label" in combined.columns:
        n_before = len(combined)
        keep = (combined["label"] >= 0).to_numpy()
        # Apply the filter PER (frame, side), not globally, and rescue any group
        # where classify rejected everything.
        #
        # classify occasionally marks every segment on one side of one frame as -1,
        # including segments 300 px long that are obviously whiskers. Filtering
        # globally then leaves that side of that frame with nothing to link, and it
        # surfaces in the labelling GUI as a frame with no labels at all and no
        # visible cause -- neighbouring frames being fine. Measured on the 840-frame
        # poke clip: 29 such frames on the left (3.5%) and 7 on the right, every one
        # with 20-39 candidates available. They are the most expensive frames to
        # correct by hand, because all three identities must be re-entered.
        #
        # A whole side going -1 in a single frame is classify failing, not evidence
        # that the whiskers left the face -- so keep that group unfiltered and let
        # the coverage model and the linker judge it.
        if keep.any() and (~keep).any():
            grp = combined.groupby(["fid", "face_side"]).indices
            rescued_rows = rescued_groups = 0
            keep = keep.copy()
            for _, idx in grp.items():
                if not keep[idx].any():
                    keep[idx] = True
                    rescued_groups += 1
                    rescued_rows += len(idx)
            combined = combined[keep].copy()
            msg = (f"[hmm_link] classify filter: kept {len(combined):,} of "
                   f"{n_before:,} detections "
                   f"({100.0*(n_before-len(combined))/n_before:.1f}% marked "
                   f"'not a whisker' by classify)")
            if rescued_groups:
                msg += (f"; rescued {rescued_rows:,} in {rescued_groups} frame-sides "
                        f"where classify rejected everything")
            print(msg)
        elif not keep.any():
            print("[hmm_link] WARNING: every detection is labelled -1; ignoring the "
                  "classify filter so there is something to link")

    if n_per_side is None:
        # Prefer classify's own answer. It assigns identities 0..N-1 per side, and
        # that count is what it decided after its own length/follicle tests -- on
        # this clip, 3 per side in every frame. estimate_n_per_side() re-derives a
        # count from per-label length statistics instead, which is strictly less
        # informed: run on the same data it returns 2, dropping a real whisker.
        n_per_side = _n_from_classify_labels(combined)
        if n_per_side:
            print(f"[hmm_link] whisker count from classify labels: {n_per_side}")
        else:
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
    return combined, apply_hmm_identity(combined, hmm)


def _default_coverage_path():
    """Path to the shipped production coverage model, if present."""
    p = os.path.join(os.path.dirname(__file__), "models", "coverage.joblib")
    return p if os.path.exists(p) else None


def _maybe_load(path):
    """Lazily load a joblib model bundle; return None on any failure (fall back)."""
    if not path or not os.path.exists(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover
        print(f"[hmm_link] could not load model {path} ({exc}); using default path.")
        return None


def link_whiskers_hmm(combined_parquet: str, wt_dir: str, base_name: str,
                      side_faces: Dict[str, str], whiskerpad=None,
                      n_per_side: Optional[Dict[str, int]] = None,
                      output_path: Optional[str] = None,
                      follicle_gate_frac: float = 0.15, follicle_max_dist: Optional[float] = None,
                      length_min_frac: float = 0.4, bridge_max_gap: int = 20,
                      angle_outlier_k: float = 4.0,
                      coverage_mode: str = "filters", identity_mode: str = "off",
                      coverage_model_path: Optional[str] = None,
                      identity_model_path: Optional[str] = None,
                      **classify_kw) -> Optional[str]:
    """End-to-end HMM linking: estimate N, reclassify chunks, stitch, join, save.

    ``side_faces`` maps face side -> the ``--face`` argument used at trace time
    (e.g. {"left": "left", "right": "right"}). ``whiskerpad`` (JSON path or dict)
    supplies per-side crop offsets so cropped measurement coordinates align with
    the full-frame combined parquet. Writes ``*_updated.parquet`` (or
    ``output_path``) and returns its path.
    """
    backbone = hmm_backbone(combined_parquet, wt_dir, base_name, side_faces,
                            whiskerpad=whiskerpad, n_per_side=n_per_side, **classify_kw)
    if backbone is None:
        return None
    combined, out = backbone

    # --- COVERAGE: learned real-vs-noise selection (add-on) OR hand-tuned filters ---
    if coverage_mode in ("model", "hybrid"):
        cov_bundle = _maybe_load(coverage_model_path or _default_coverage_path())
    else:
        cov_bundle = None
    if cov_bundle is not None:
        from . import coverage_model as _cov
        if coverage_mode == "hybrid" and length_min_frac:
            out = filter_length_outliers(out, length_min_frac)
        gate = follicle_max_dist if follicle_max_dist else estimate_follicle_gate(out, follicle_gate_frac)
        before = len(out)
        out = _cov.apply_coverage_model(out, combined, cov_bundle, side_faces=side_faces, gate_px=gate)
        print(f"[hmm_link] coverage model: {before} -> {len(out)} detections.")
    else:
        if length_min_frac:
            before = len(out)
            out = filter_length_outliers(out, length_min_frac)
            print(f"[hmm_link] length-outlier filter dropped {before - len(out)} detections.")
        # Empirical follicle gate (scales with camera/zoom via whisker length).
        gate = follicle_max_dist if follicle_max_dist else estimate_follicle_gate(out, follicle_gate_frac)
        print(f"[hmm_link] follicle gate = {gate:.1f} px")
        # Trust whisk's HMM identity for ambiguity resolution; only *remove* clear
        # noise (a detection far from its identity's base, e.g. a cotton strand). We do
        # NOT re-assign identities by position -- that flickers ("christmas tree") and
        # can invert whole frames, undoing whisk's temporally-modelled identity.
        before = len(out)
        out = filter_follicle_outliers(out, gate)
        print(f"[hmm_link] follicle-outlier filter dropped {before - len(out)} detections.")
        if bridge_max_gap:
            before = len(out)
            out = bridge_gaps(out, combined, max_gap=bridge_max_gap, gate_px=gate,
                              min_length_frac=length_min_frac or 0.4)
            print(f"[hmm_link] gap-bridging recovered {len(out) - before} detections.")
        if angle_outlier_k:
            before = len(out)
            out = filter_angle_outliers(out, angle_k=angle_outlier_k)
            print(f"[hmm_link] angle-outlier filter dropped {before - len(out)} detections.")

    # --- IDENTITY: learned conservative re-ranker (add-on) ---
    # "rerank" loads a pre-trained per-session model; "bootstrap" trains one on this clip's
    # own HMM-confident runs (no labels) -- the self-contained deployment path.
    if identity_mode in ("rerank", "bootstrap"):
        from . import identity_model as _idm
        if identity_mode == "bootstrap":
            id_bundle = _idm.train_identity(out, gt=None)
        else:
            id_bundle = _maybe_load(identity_model_path)
        if id_bundle is not None and id_bundle.get("models"):
            # WW_RERANK_W_ANGLE exists so the angle-continuity term can be A/B'd on a
            # fixed set of detections: linking is cheap to re-run, tracing is not, and
            # comparing two full pipeline runs confounds the term with everything else
            # that differs between them. 0 reproduces the follicle-only behaviour.
            w_angle = float(os.environ.get("WW_RERANK_W_ANGLE", "1.0"))
            out = _idm.rerank_identity(out, id_bundle, side_faces=side_faces,
                                       w_angle=w_angle)
            print(f"[hmm_link] identity re-ranker applied ({identity_mode}), "
                  f"w_angle={w_angle}.")

    out = out.sort_values(["fid", "wid"])
    output_path = output_path or combined_parquet.replace(".parquet", "_updated.parquet")
    out.to_parquet(output_path)
    print(f"[hmm_link] saved {len(out)} rows to {output_path}")
    return output_path
