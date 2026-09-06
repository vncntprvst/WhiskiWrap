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
import time as _time
from concurrent.futures import ThreadPoolExecutor
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


_T0 = [None]


def _stage(msg: str) -> None:
    """Timestamped progress line, FLUSHED.

    Every print in this module was block-buffered: redirected to a file (which is
    what a batch job is), Python holds ~8 KB before writing, so a run that has been
    going for an hour shows an empty log and a stage that never reports is
    indistinguishable from one that is hung. That is precisely how an 8.6 h stall
    went unnoticed. Timing each stage also means the next regression shows up as a
    number rather than as a job that mysteriously hits the wall.
    """
    import time as _time
    now = _time.time()
    if _T0[0] is None:
        _T0[0] = now
    print(f"[hmm_link +{now - _T0[0]:7.1f}s] {msg}", flush=True)


def _reclassify_workers() -> int:
    """How many chunks to reclassify at once.

    Read from the scheduler rather than guessed: under SLURM the process is bound
    to `--cpus-per-task` cores, and `os.cpu_count()` reports the whole NODE, so
    trusting it would oversubscribe an allocation by up to 4x on a shared node.
    WT_RECLASSIFY_WORKERS overrides for the local case.
    """
    env = os.environ.get("WT_RECLASSIFY_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            return max(1, int(slurm))
        except ValueError:
            pass
    try:
        # the cores this process may actually run on, not the machine's total
        return max(1, len(os.sched_getaffinity(0)))       # Linux only
    except AttributeError:
        return max(1, (os.cpu_count() or 2) // 2)


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
            _stage(f"{side}: rescued {len(rescued)} persistent long-peaking "
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
                _stage(f"{side}: n={out[side]}; rejected but persistent: {desc}"
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


def _seam_sig(d: pd.DataFrame, which: str, k_pos: int = 25, k_ang: int = 5
              ) -> pd.DataFrame:
    """Per-state signature at a chunk seam: follicle, angle and length.

    The two quantities need different windows. A follicle barely moves, so a median
    over ~25 frames is a stable estimate of where the whisker is rooted. An ANGLE
    sweeps several degrees over that many frames -- on this data a whisker moves
    ~1.5 deg per frame -- so a 25-frame median describes the middle of a sweep and
    not the angle at the seam. Angle is therefore taken from only the few frames
    adjacent to the boundary, which is what has to be continuous across it.
    """
    out = {}
    for st, g in d.groupby("state"):
        g = g.sort_values("fid")
        gp = g.tail(k_pos) if which == "end" else g.head(k_pos)
        ga = g.tail(k_ang) if which == "end" else g.head(k_ang)
        out[st] = dict(
            follicle_x=float(gp["follicle_x"].median()),
            follicle_y=float(gp["follicle_y"].median()),
            angle=float(ga["angle"].median()) if "angle" in g else np.nan,
            length=float(gp["length"].median()) if "length" in g else np.nan,
        )
    return pd.DataFrame(out).T


def _sig_cost(a: dict, b: dict, pos_scale: float, ang_scale: float,
              use_angle: bool = True, len_scale: float = 40.0) -> float:
    """Normalised distance between two seam signatures.

    Position ALONE cannot separate whiskers on one pad -- their bases sit within a
    few px of each other, which is the same failure the identity re-ranker had
    before angle continuity was added to it. On the hand-corrected clip the right
    side's bases are at follicle_y 186.5 and 191.8, i.e. 5 px apart, inside the
    10 px confidence margin, so every seam match there was ambiguous and fell back
    to whole-chunk means. Angle separates them: those same two whiskers sit ~16 deg
    apart.

    ANGLE ONLY APPLIES ACROSS AN ADJACENT SEAM (``use_angle``)
        Angle continuity is a statement about consecutive frames. Across a gap it
        says nothing: a whisker sweeping ~2 deg per frame has moved through several
        hundred degrees over a hundred-frame absence, so a stored angle is not
        merely stale but arbitrary. Matching a RETURNING whisker on it swapped
        identities in 2 of 6 seeds of the absence test while the id COUNT stayed
        correct -- an error no count-based check would have caught.

        So angle discriminates for a track last seen in the immediately preceding
        chunk; for one returning after an absence it is dropped, and length carries
        the shape information instead. Position is the quantity that stays valid
        across a gap.
    """
    d = np.hypot(a["follicle_x"] - b["follicle_x"], a["follicle_y"] - b["follicle_y"])
    c = d / pos_scale
    if use_angle and np.isfinite(a.get("angle", np.nan)) \
            and np.isfinite(b.get("angle", np.nan)):
        # wrap to +-180 so a whisker near the +-180 boundary is not seen as far
        da = abs(((a["angle"] - b["angle"] + 180.0) % 360.0) - 180.0)
        c += da / ang_scale
    elif np.isfinite(a.get("length", np.nan)) and np.isfinite(b.get("length", np.nan)):
        c += abs(a["length"] - b["length"]) / len_scale
    return float(c)


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
                            axis: str = "follicle_y", boundary_frames: int = 5,
                            confident_margin: float = 10.0,
                            angle_frames: int = 1,
                            pos_scale: float = 10.0, ang_scale: float = 15.0,
                            gate: float = float("inf")) -> pd.DataFrame:
    """Assign a global identity (``gid``) across chunks.

    ``chunks`` is a list of ``(chunk_start, df)`` ordered by chunk_start, where each
    df contains only identified detections (``state`` >= 0) with a global ``fid``.

    WHAT WAS WRONG WITH MATCHING ON POSITION ALONE
        Whiskers on one pad have bases a few px apart -- on the hand-corrected clip
        the right side sits at follicle_y 186.5 and 191.8 -- so a positional cost
        cannot tell them apart, and the old code's 10 px confidence margin was never
        satisfied. It then fell back to whole-chunk means, which during whisking are
        worse still. Measured against known truth, that swapped two whiskers at a
        chunk boundary and split one whisker into two identities.

        Angle separates them: those two whiskers are ~16 deg apart. This is the same
        fix the identity re-ranker already had, for the same reason.

    THE ANGLE MUST COME FROM THE FRAME AT THE SEAM
        A seam is between two ADJACENT frames. Whiskers protract at ~1.5-3 deg per
        frame, so a median over even five frames either side differs by ~10 deg for
        the SAME whisker -- more than the ~4 deg separating two whiskers. Measured:
        identity accuracy 98.5% with a one-frame angle window, 86.7% at two frames,
        69.9% at five. Position is noisier per frame and is still smoothed.

    IDENTITIES PERSIST ACROSS ABSENCES
        The old code compared only against the immediately preceding chunk, so a
        whisker missing from one chunk could never rejoin its own identity: it was
        unmatched, minted a fresh gid, and the reference set grew monotonically.
        Over a thousand chunks that is how a handful of whiskers became 60, 129 or
        678 global ids. Tracks are kept here and matched against their last-seen
        signature, or their long-run one when they have been away.

    ``gate`` IS OFF BY DEFAULT, AND THAT WAS MEASURED
        Hungarian always returns a full assignment, so a distance gate looks like
        the right way to stop a spurious detection taking a real whisker's
        identity. Tuned on an 840-frame clip -- FOUR seams -- a gate of 5.0 looked
        best. On a real 3787-seam session it was a disaster: it rejected legitimate
        matches deep in the cost distribution's tail and split 3 whiskers into 17
        identities, where the ungated version returns exactly 3.

        Measured on sc014_0324_001 right side, 3787 chunks:

            n=3 (the animal's real count)      n=12 (what the pipeline estimates)
            old            3 ids, 100%         old           200 ids, top 35%
            gate=5        17 ids, top 54%      gate=5         99 ids, top 59%
            gate=20        3 ids, 100%         gate=20        21 ids, top 81%
            gate=off       3 ids, 100%         gate=off       12 ids, top 93%

        So it stays as a parameter and defaults to off. A constant tuned on four
        samples has no business gating three thousand.

    Measured two ways. On the hand-corrected clip with simulated chunking
    (tests/test_stitching.py), across dropped-whisker and spurious-detection rates
    from 0 to 0.4: worst-case identity accuracy 72.9% -> 88.6%, mean 86.5% -> 96.6%,
    clean 100% both. And on a real 3787-chunk session (above), where the persistent
    tracks are what matter: 200 identities -> 12, none persistent -> all at 93%.

    NOTE the 12 is `n`, not the animal's whisker count. This side has ~3 whiskers;
    the pipeline asked classify for 12, so 9 of those identities are fur tracked
    consistently. Stitching is then doing its job correctly on wrong input -- the
    count estimate is a separate defect, upstream of here.
    """
    chunks = sorted(chunks, key=lambda t: t[0])
    out = []
    # gid -> persistent track. `end` is its signature where it was last seen, `mean`
    # its long-run signature, `last` the chunk index it was last seen in.
    tracks: Dict[int, dict] = {}
    next_gid = 0

    for ci, (_, d) in enumerate(chunks):
        if d.empty:
            continue
        start = _seam_sig(d, "start", boundary_frames, angle_frames)
        end = _seam_sig(d, "end", boundary_frames, angle_frames)
        states = list(start.index)

        if not tracks:
            mapping = {st: i for i, st in enumerate(start[axis].sort_values().index)}
            next_gid = len(mapping)
        else:
            # MATCH AGAINST EVERY KNOWN IDENTITY, not just the previous chunk.
            #
            # The old code compared only against the chunk immediately before, so a
            # whisker missing from one chunk could never rejoin its own identity --
            # it was unmatched, minted a fresh gid, and the reference set grew. That
            # is what turns a handful of whiskers into 60 or 678 global ids over a
            # thousand chunks. A whisker that vanishes for a while and comes back is
            # the normal case, not an exception, so identities persist here and are
            # matched against their last-seen signature (or their long-run one when
            # they have been away).
            gids = list(tracks.keys())
            cost = np.empty((len(gids), len(states)), float)
            for gi, g in enumerate(gids):
                t = tracks[g]
                adjacent = (t["last"] == ci - 1)
                ref = t["end"] if adjacent else t["mean"]
                for si, st in enumerate(states):
                    cost[gi, si] = _sig_cost(ref, start.loc[st].to_dict(),
                                             pos_scale, ang_scale,
                                             use_angle=adjacent)
            ri, cix = linear_sum_assignment(cost)
            mapping = {}
            for r, c in zip(ri, cix):
                # A pair is accepted only if it is actually close. Hungarian always
                # returns a full assignment, so without a gate a spurious detection
                # is guaranteed to steal some real whisker's identity -- which is
                # exactly the swap seen at frame 400 on the hand-checked clip.
                if cost[r, c] <= gate:
                    mapping[states[c]] = gids[r]
            for st in states:
                if st not in mapping:
                    mapping[st] = next_gid
                    next_gid += 1

        dd = d.copy()
        dd["gid"] = dd["state"].map(mapping)
        out.append(dd)

        for st in states:
            g = mapping[st]
            e = end.loc[st].to_dict()
            s = start.loc[st].to_dict()
            if g in tracks:
                t = tracks[g]
                n = t["n"]
                # running mean over the frames this identity has been seen in
                t["mean"] = {k: (t["mean"][k] * n + s[k]) / (n + 1)
                             if np.isfinite(s.get(k, np.nan))
                             and np.isfinite(t["mean"].get(k, np.nan))
                             else t["mean"].get(k, s.get(k))
                             for k in ("follicle_x", "follicle_y", "angle", "length")}
                t["n"] = n + 1
                t["end"] = e
                t["last"] = ci
            else:
                tracks[g] = dict(end=e, mean=dict(s), n=1, last=ci)

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
    files = []
    for meas in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d{8})\.measurements$", os.path.basename(meas))
        if m:
            files.append((int(m.group(1)), meas))

    # Each chunk is two whisk subprocesses against one file, in place, with no
    # shared state -- so this is embarrassingly parallel, and it was running one
    # chunk at a time on a 32-core node. A real session has ~1100 chunks per side;
    # at even half a second each that is ten minutes of a single core while the
    # rest idle. Threads (not processes) because the time is spent inside
    # `subprocess.run`, which releases the GIL, and results come back without
    # pickling a DataFrame per chunk.
    #
    # ThreadPoolExecutor.map preserves input order, which matters: stitching walks
    # chunks in ascending chunk_start and would otherwise stitch them shuffled.
    def _one(item):
        chunk_start, meas = item
        reclassify_measurements(meas, face, n, **classify_kw)
        d = read_measurements(meas)
        d = d[d["state"] >= 0].copy()
        d["fid"] = d["fid"] + chunk_start
        d["follicle_x"] = d["follicle_x"] + ox
        d["follicle_y"] = d["follicle_y"] + oy
        return (chunk_start, d)

    workers = _reclassify_workers()
    _stage(f"{side}: reclassifying {len(files)} chunks on {workers} worker(s)")
    if workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            chunks = list(ex.map(_one, files))
    else:
        chunks = [_one(f) for f in files]
    if not chunks:
        return pd.DataFrame()
    _stage(f"{side}: reclassify done; stitching {len(chunks)} chunks")
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
    # Two things here were quadratic-ish and both mattered on a real session.
    #
    #   * `hmm[(hmm.fid == fid) & (hmm.face_side == side)]` rebuilt a boolean mask
    #     over the WHOLE hmm table once per (frame, side). At 220k frames that is
    #     ~440k scans of a million rows. This is the same mistake that made
    #     rerank_identity an 8-hour stage; `groupby(...).indices` is one pass.
    #   * Accumulating `c.iloc[r].copy()` built one pandas Series per matched
    #     detection -- about a million short-lived objects -- before
    #     `pd.DataFrame(keep)` glued them back together.
    #
    # `.indices` gives POSITIONAL arrays, so none of this depends on the frames
    # carrying a unique or aligned index.
    if combined.empty or hmm.empty:
        return pd.DataFrame(columns=combined.columns)

    c_groups = combined.groupby(["fid", "face_side"], sort=False).indices
    h_groups = hmm.groupby(["fid", "face_side"], sort=False).indices
    cf = combined[["follicle_x", "follicle_y"]].to_numpy(float)
    hf = hmm[["follicle_x", "follicle_y"]].to_numpy(float)
    hgid = hmm["gid"].to_numpy()

    # Sorted key order, because that is what `groupby(...)` iteration gave before
    # and the output row order is observable downstream.
    keep_pos, keep_gid = [], []
    for key in sorted(c_groups):
        cpos = c_groups[key]
        hpos = h_groups.get(key)
        if hpos is None or len(hpos) == 0:
            continue
        cx, hx = cf[cpos], hf[hpos]
        dist = np.sqrt(((cx[:, None, :] - hx[None, :, :]) ** 2).sum(-1))
        ci, hi = linear_sum_assignment(dist)
        ok = dist[ci, hi] <= gate_px
        if not ok.any():
            continue
        keep_pos.append(cpos[ci[ok]])
        keep_gid.append(hgid[hpos[hi[ok]]])

    if not keep_pos:
        return pd.DataFrame(columns=combined.columns)
    out = combined.iloc[np.concatenate(keep_pos)].copy()
    out["label"] = out["wid"].to_numpy()
    out["wid"] = np.concatenate(keep_gid).astype(int)
    return out


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
    # This walks frames in order and cannot be vectorised across them -- each frame's
    # prediction depends on the previous one. What it can stop doing is rebuilding
    # the frame's rows with `sd[sd["fid"] == fid]`, a full boolean scan of the side's
    # table once per frame, and materialising a pandas Series per matched detection.
    # Positions are grouped once, and the per-identity state is a small array rather
    # than dicts of `cen.loc[w]` lookups.
    cen = df.groupby("wid")[["follicle_x", "follicle_y"]].median()
    id_side = df.groupby("wid")["face_side"].first()
    fol = df[["follicle_x", "follicle_y"]].to_numpy(float)
    fid_all = df["fid"].to_numpy()
    side_all = df["face_side"].to_numpy()

    keep_pos, keep_wid = [], []
    for side in df["face_side"].unique():
        ids = [w for w in cen.index if id_side[w] == side]
        if not ids:
            continue
        ids_arr = np.asarray(ids)
        cen_arr = cen.loc[ids].to_numpy(float)
        last = cen_arr.copy()
        missed = np.zeros(len(ids), dtype=int)

        pos = np.flatnonzero(side_all == side)
        order = np.argsort(fid_all[pos], kind="stable")   # stable: keeps within-frame order
        pos = pos[order]
        bounds = np.flatnonzero(np.diff(fid_all[pos])) + 1
        for grp in np.split(pos, bounds):
            if grp.size == 0:
                continue
            G = fol[grp]
            P = np.where((missed <= max_missed)[:, None], last, cen_arr)
            D = np.sqrt(((G[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            gi, ci = linear_sum_assignment(D)
            ok = D[gi, ci] <= gate_px
            if ok.any():
                mg, mc = gi[ok], ci[ok]
                keep_pos.append(grp[mg])
                keep_wid.append(ids_arr[mc])
                last[mc] = G[mg]
                missed[mc] = 0
                unmatched = np.setdiff1d(np.arange(len(ids)), mc, assume_unique=False)
            else:
                unmatched = np.arange(len(ids))
            missed[unmatched] += 1

    if not keep_pos:
        return pd.DataFrame(columns=df.columns)
    out = df.iloc[np.concatenate(keep_pos)].copy()
    out["wid"] = np.concatenate(keep_wid).astype(int)
    return out


def reassign_by_centroid(df: pd.DataFrame, gate_px: float) -> pd.DataFrame:
    """Per frame, assign each detection to the nearest identity centroid (Hungarian).

    Corrects within-frame identity swaps (e.g. whisker 1 mislabelled 2 when whisker
    2 is occluded) using each identity's robust (median) follicle centroid, and
    drops detections farther than ``gate_px`` from every centroid (noise). One-to-one
    per (frame, side); occluded identities simply go unmatched (a clean gap).
    """
    if df.empty:
        return df
    # The per-side identity list and its centroid matrix do not depend on the frame,
    # but were rebuilt inside the loop: `[w for w in cen.index if id_side[w] == side]`
    # is a Series lookup per identity per frame, which on a 220k-frame session is
    # millions of them before any geometry is computed. Hoisted, and the output is
    # assembled once instead of one Series per matched detection.
    cen = df.groupby("wid")[["follicle_x", "follicle_y"]].median()
    id_side = df.groupby("wid")["face_side"].first()
    per_side = {}
    for side in df["face_side"].unique():
        ids = [w for w in cen.index if id_side[w] == side]
        if ids:
            per_side[side] = (np.asarray(ids), cen.loc[ids].to_numpy(float))

    fol = df[["follicle_x", "follicle_y"]].to_numpy(float)
    groups = df.groupby(["fid", "face_side"], sort=False).indices

    keep_pos, keep_wid = [], []
    for key in sorted(groups):
        side = key[1]
        got = per_side.get(side)
        if got is None:
            continue
        ids_arr, C = got
        grp = groups[key]
        G = fol[grp]
        D = np.sqrt(((G[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        gi, ci = linear_sum_assignment(D)
        ok = D[gi, ci] <= gate_px
        if not ok.any():
            continue
        keep_pos.append(grp[gi[ok]])
        keep_wid.append(ids_arr[ci[ok]])

    if not keep_pos:
        return pd.DataFrame(columns=df.columns)
    out = df.iloc[np.concatenate(keep_pos)].copy()
    out["wid"] = np.concatenate(keep_wid).astype(int)
    return out


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
                gate_px: float = 25.0, min_length_frac: float = 0.4,
                rescue_length_frac: float = 0.10,
                rescue_gate_frac: float = 0.32) -> pd.DataFrame:
    """Recover identities for frames where a *visible* whisker was left unclassified.

    For each identity, short gaps (<= ``max_gap`` frames) between present frames are
    filled by searching the combined detections for an as-yet-unassigned, long
    enough whisker near the interpolated follicle position. This recovers whiskers
    that whisk failed to classify (``state = -1``) without inventing a whisker for
    genuinely occluded frames (no suitable detection exists there).

    PARTIAL OCCLUSION
        A whisker crossed by the cue tip is not gone: it is covered part-way along
        and still traced, just short. ``min_length_frac`` then rejects it -- the
        gate excludes the very case this function exists for. Measured against a
        hand-corrected clip, the whiskers a human had to add back were a median
        0.35 of their identity's normal length, against 1.00 for the rows the
        pipeline kept, and 88% of the remaining gaps had a traced candidate sitting
        unused. So they are found; they are discarded here.

        The rescue is a strict SECOND pass, tried only where the normal rule found
        nothing. Simply lowering the floor made things worse elsewhere: adoption is
        greedy on base distance, so a short wrong candidate would win and block the
        correct longer one, costing 8 real whiskers on a fast-whisking clip. As a
        fallback it cannot displace anything the current rule would have found.

        The gate for a rescued segment is its BASE position, tightly -- a third of
        the normal follicle gate, expressed as a fraction because that gate is
        itself estimated per clip and scales with camera and zoom. The cue tip
        covers the far end of a whisker, not its root, so a partially occluded
        whisker's follicle is still visible and lands almost exactly where
        interpolation predicts -- which is the property that separates it from
        noise, and the thing the length floor was standing in for.

        Measured over 7 hand-corrected clips: 352 -> 395 detections recovered for
        14 -> 21 not in the human's labels; on the cue-poke occlusion clip 19 -> 46
        for one extra. Set ``rescue_length_frac`` to 0 to disable.
    """
    if out.empty:
        return out
    # Membership is a POSITIONAL BOOLEAN MASK, not a Python set.
    #
    # `_fr.index.isin(assigned)` looks like a cheap set test and is not: pandas
    # builds a hash table over the whole `assigned` collection on every call, and
    # `assigned` holds every already-linked detection -- over a million of them on
    # a real session. Called once per gap frame per whisker, that is the same
    # quadratic shape as the other three stages, just wearing a set.
    #
    # Falls back to the original set when `combined` has a non-unique index, since
    # positions cannot then be recovered from labels.
    fast = combined.index.is_unique
    if fast:
        pos_of = pd.Series(np.arange(len(combined)), index=combined.index)
        take = pos_of.reindex(out.index)
        is_assigned = np.zeros(len(combined), dtype=bool)
        is_assigned[take.dropna().to_numpy(dtype=np.int64)] = True
        fol_x = combined["follicle_x"].to_numpy(float)
        fol_y = combined["follicle_y"].to_numpy(float)
        len_arr = combined["length"].to_numpy(float)
        pos_groups = combined.groupby(["face_side", "fid"], sort=False).indices
    assigned = set(out.index)
    new_rows = []
    adopted_pos = []
    for side in out["face_side"].unique():
        cs = combined[combined["face_side"] == side]
        # Same O(rows x frames) trap as the identity re-ranker: `cs[cs["fid"] == f]`
        # per gap frame rescans the whole side. Index once.
        cs_by_fid = None if fast else {int(f): g for f, g in cs.groupby("fid", sort=False)}
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

                    if fast:
                        fr_pos = pos_groups.get((side, int(f)))
                        if fr_pos is None:
                            continue
                        free_pos = fr_pos[~is_assigned[fr_pos]]

                        def _adopt(cand, limit):
                            if cand.size == 0:
                                return None
                            dd = np.hypot(fol_x[cand] - ex, fol_y[cand] - ey)
                            k = int(dd.argmin())        # first minimum, as idxmin gave
                            return None if dd[k] > limit else int(cand[k])

                        pos = _adopt(
                            free_pos[len_arr[free_pos] >= min_length_frac * med_len],
                            gate_px)
                        if pos is None and rescue_length_frac:
                            # partial occlusion: short, but its base must be where the
                            # interpolation says (see the note in the docstring)
                            pos = _adopt(
                                free_pos[len_arr[free_pos] >= rescue_length_frac * med_len],
                                rescue_gate_frac * gate_px)
                        if pos is not None:
                            adopted_pos.append((pos, int(wid)))
                            is_assigned[pos] = True
                        continue

                    _fr = cs_by_fid.get(int(f))
                    if _fr is None:
                        continue
                    free = _fr[~_fr.index.isin(assigned)]

                    def _adopt(pool, limit):
                        if pool.empty:
                            return None
                        dd = np.hypot(pool["follicle_x"] - ex,
                                      pool["follicle_y"] - ey)
                        if dd.min() > limit:
                            return None
                        return dd.idxmin()

                    idx = _adopt(free[free["length"] >= min_length_frac * med_len],
                                 gate_px)
                    if idx is None and rescue_length_frac:
                        idx = _adopt(
                            free[free["length"] >= rescue_length_frac * med_len],
                            rescue_gate_frac * gate_px)
                    if idx is not None:
                        row = cs.loc[idx].copy()
                        row["label"] = row["wid"]
                        row["wid"] = int(wid)
                        new_rows.append(row)
                        assigned.add(idx)

    if adopted_pos:
        pos = np.fromiter((p for p, _ in adopted_pos), dtype=np.int64,
                          count=len(adopted_pos))
        wids = np.fromiter((w for _, w in adopted_pos), dtype=np.int64,
                           count=len(adopted_pos))
        add = combined.iloc[pos].copy()
        add["label"] = add["wid"].to_numpy()
        add["wid"] = wids
        out = pd.concat([out, add])
    elif new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)])
    return out


def _labels_look_like_identities(combined, frac: float = 0.6) -> bool:
    """Was classify run with a real ``-n N``, or in automatic per-segment mode?

    In automatic mode (``-n -1``, which is what the tracing pipeline passes) classify
    gives nearly every segment its own label, so the number of DISTINCT labels in a
    frame approaches the number of detections in it. With a real N it is about N.

    Measured on the ground-truth clips, whose combined parquets come from automatic
    mode: 18-19 distinct labels per frame against ~18 detections per frame, i.e. a
    ratio near 1. A genuine 3-whisker labelling would sit near 3/18.
    """
    if "label" not in combined.columns or "face_side" not in combined.columns:
        return False
    for _, g in combined.groupby("face_side"):
        lab = g[g["label"] >= 0]
        if lab.empty:
            continue
        per_frame_labels = lab.groupby("fid")["label"].nunique().median()
        per_frame_dets = g.groupby("fid").size().median()
        if per_frame_dets and per_frame_labels >= frac * per_frame_dets:
            return False
    return True


def _n_from_classify_labels(combined, min_frame_frac: float = 0.5) -> Dict[str, int]:
    """Whiskers per side, taken from the identities classify assigned.

    classify labels each segment -1 (not a whisker) or 0,1,2... (whisker n), so
    max(label)+1 is its answer -- BUT ONLY IF IT WAS GIVEN A REAL ``-n N``.

    The tracing pipeline passes `num_whiskers = -1` (automatic), and in that mode
    classify gives nearly every segment its own label. max(label)+1 then counts
    segments: against hand-edited ground truth where the answer is 3 whiskers per
    side, this returns 28-37, and on real sessions 9-13. Callers must therefore
    check `_labels_look_like_identities` first; `hmm_backbone` does.

    The consequences of not checking are severe and were both observed: a moderate
    overcount (n=12) lets classify succeed but makes 9 of 12 identities fur, which
    is where 678 global identities came from; a large one (n=28) makes classify
    return no identities at all, so HMM linking silently produces nothing and the
    pipeline falls back to the geometry linker without saying so.

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
            _stage(f"{side}: identity present in few frames "
                  f"(kept anyway -- occlusion is not absence): {thin}")
    return out


def hmm_backbone(combined_parquet, wt_dir, base_name, side_faces, *,
                 whiskerpad=None, n_per_side=None, classify_filter=True,
                 **classify_kw):
    """Run the whisk-HMM backbone (classify+reclassify per chunk, stitch, join) and
    return ``(combined_df, out_df)`` where ``out`` is post-``apply_hmm_identity`` (the raw
    candidate labeling, before any coverage filtering / identity re-rank). Returns None if
    no identities were produced. Exposed so the autotune loop can cache this expensive step
    once and apply learned add-ons offline."""
    combined = pd.read_parquet(combined_parquet)

    # Compute the shape descriptors ONCE, here, and let them travel.
    #
    # Everything downstream is derived from these rows -- apply_hmm_identity returns
    # `combined.iloc[...]`, bridge_gaps adopts more of the same rows, follicle_snap
    # copies -- so columns added now are present at every later stage. Without this
    # the coverage model, the identity bootstrap and the identity re-ranker each
    # rebuild them from scratch at ~38 us per row, which is most of the link.
    try:
        from . import detection_features as _dF
        if not all(c in combined.columns for c in _dF._SHAPE_SCALAR_COLS):
            _t0 = _time.time()
            combined = _dF.add_shape_features(combined, include_vec=False)
            _stage(f"shape features for {len(combined)} detections "
                   f"({_time.time() - _t0:.1f}s, computed once for the whole link)")
    except Exception as exc:                                  # noqa: BLE001
        # Not fatal: every consumer still computes them itself if they are absent.
        _stage(f"could not precompute shape features ({exc}); each stage will "
               f"recompute them")

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
    # WW_CLASSIFY_FILTER=0 keeps every detection and lets the coverage model and
    # the per-side count decide instead. Measured against three hand-corrected
    # hard-case clips: the pipeline missed 40.8% of the whiskers a human labelled,
    # every one of them traced and then discarded, and 62% of those had been marked
    # -1 by classify. A signal used as a veto costs more than it earns -- the same
    # conclusion the whiskerness map forced. The count still comes from the labels,
    # which is what they are reliably good for.
    _cf = os.environ.get("WW_CLASSIFY_FILTER")
    if _cf is not None:
        classify_filter = _cf != "0"
    if "label" in combined.columns and classify_filter:
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
        # WHICH ESTIMATE TO TRUST, AND WHY IT IS NO LONGER THE LABELS
        #
        # `_n_from_classify_labels` reads max(label)+1, on the premise that classify
        # already decided the whisker count using its own length and follicle tests.
        # That premise requires classify to have been given a real `-n N`. The
        # tracing pipeline passes `num_whiskers = -1` (automatic), and in that mode
        # classify labels essentially EVERY segment separately -- so max(label)+1
        # counts segments, not whiskers.
        #
        # Measured against the two hand-edited ground-truth clips, where the answer
        # is 3 whiskers per side in all four cases:
        #
        #     estimate_n_per_side       {left: 3,  right: 3}   {left: 3,  right: 3}
        #     _n_from_classify_labels   {left: 36, right: 37}  {left: 28, right: 27}
        #
        # Those inflated counts are the origin of the 60/129/678 global identities
        # seen on real sessions: forcing classify to find 12 whiskers on a 3-whisker
        # side makes 9 of them fur, and stitching then has 9 spurious states per
        # chunk to contend with.
        #
        # The labels are still used when they look like real identities. The tell is
        # that automatic mode gives nearly as many distinct labels per FRAME as
        # there are detections in it, whereas a genuine `-n N` run gives about N.
        n_lab = _n_from_classify_labels(combined)
        n_per_side = estimate_n_per_side(combined)
        if n_lab and _labels_look_like_identities(combined):
            _stage(f"whisker count from classify labels: {n_lab}")
            n_per_side = n_lab
        elif n_lab:
            _stage(f"classify labels look per-segment, not per-whisker "
                   f"({n_lab}); using the length-based estimate {n_per_side} instead")
    offsets = _side_offsets(whiskerpad) if whiskerpad is not None else {}
    _stage(f"whiskers per side: {n_per_side}")
    hmm_parts = []
    base = 0
    for side in sorted(side_faces):
        n = n_per_side.get(side, 1)
        part = relabel_side_with_hmm(wt_dir, base_name, side, side_faces[side], n,
                                     coord_offset=offsets.get(side, (0.0, 0.0)), **classify_kw)
        if part.empty:
            _stage(f"{side}: no identities produced.")
            continue
        part["gid"] = part["gid"] + base                 # keep sides disjoint
        base += part["gid"].nunique()
        hmm_parts.append(part)
        _stage(f"{side}: {part['gid'].nunique()} whiskers, {len(part)} detections.")
    if not hmm_parts:
        return None
    hmm = pd.concat(hmm_parts, ignore_index=True)
    _stage(f"joining {len(hmm)} hmm detections onto {len(combined)} combined rows")
    joined = apply_hmm_identity(combined, hmm)
    _stage(f"join produced {len(joined)} rows")
    return combined, joined


def _default_coverage_path():
    """Path to the coverage model to use, if present.

    WW_COVERAGE_MODEL overrides it, by absolute path or by bare filename resolved
    against models/. That exists so a retrained model can be A/B'd against the
    shipped one without editing code or swapping files: the shipped
    coverage.joblib is what every earlier result was produced with, and replacing
    it in place would make old and new runs quietly incomparable.
    """
    env = os.environ.get("WW_COVERAGE_MODEL")
    if env:
        cand = env if os.path.isabs(env) else os.path.join(
            os.path.dirname(__file__), "models", env)
        if os.path.exists(cand):
            return cand
        _stage(f"WW_COVERAGE_MODEL={env!r} not found; "
              f"falling back to the shipped model")
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
        _stage(f"could not load model {path} ({exc}); using default path.")
        return None


def link_whiskers_hmm(combined_parquet: str, wt_dir: str, base_name: str,
                      side_faces: Dict[str, str], whiskerpad=None,
                      n_per_side: Optional[Dict[str, int]] = None,
                      output_path: Optional[str] = None,
                      follicle_gate_frac: float = 0.15, follicle_max_dist: Optional[float] = None,
                      length_min_frac: float = 0.4, bridge_max_gap: int = 20,
                      angle_outlier_k: float = 4.0,
                      coverage_mode: str = "filters", identity_mode: str = "off",
                      classify_filter: bool = True, follicle_snap: bool = True,
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
    _stage(f"linking {os.path.basename(combined_parquet)}")
    backbone = hmm_backbone(combined_parquet, wt_dir, base_name, side_faces,
                            whiskerpad=whiskerpad, n_per_side=n_per_side,
                            classify_filter=classify_filter, **classify_kw)
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
        _stage(f"coverage model: {before} -> {len(out)} detections.")
    else:
        if length_min_frac:
            before = len(out)
            out = filter_length_outliers(out, length_min_frac)
            _stage(f"length-outlier filter dropped {before - len(out)} detections.")
        # Empirical follicle gate (scales with camera/zoom via whisker length).
        gate = follicle_max_dist if follicle_max_dist else estimate_follicle_gate(out, follicle_gate_frac)
        _stage(f"follicle gate = {gate:.1f} px")
        # Trust whisk's HMM identity for ambiguity resolution; only *remove* clear
        # noise (a detection far from its identity's base, e.g. a cotton strand). We do
        # NOT re-assign identities by position -- that flickers ("christmas tree") and
        # can invert whole frames, undoing whisk's temporally-modelled identity.
        before = len(out)
        out = filter_follicle_outliers(out, gate)
        _stage(f"follicle-outlier filter dropped {before - len(out)} detections.")
        if angle_outlier_k:
            before = len(out)
            out = filter_angle_outliers(out, angle_k=angle_outlier_k)
            _stage(f"angle-outlier filter dropped {before - len(out)} detections.")

    # --- GAP BRIDGING: applies to BOTH coverage paths ---
    # This used to sit inside the hand-tuned-filters branch above, so once the
    # coverage model became the default it stopped running entirely -- silently,
    # because nothing downstream reports a gap that was never filled. Measured
    # against 7 hand-corrected clips, restoring it to the model path recovers 352
    # detections the live pipeline was leaving on the floor, and 395 with the
    # partial-occlusion rescue.
    #
    # It belongs after coverage selection, not instead of it: the coverage model
    # decides real-vs-noise on a detection's own appearance, while this asks a
    # different question -- is there an unassigned detection exactly where this
    # identity must be, given where it was before and after. A whisker the model
    # rejected for being stubby is precisely the one this should get back.
    if bridge_max_gap:
        before = len(out)
        out = bridge_gaps(out, combined, max_gap=bridge_max_gap, gate_px=gate,
                          min_length_frac=length_min_frac or 0.4)
        _stage(f"gap-bridging recovered {len(out) - before} detections.")

    # --- BASE RECONSTRUCTION for paw-occluded whiskers ---
    # Adds base_occluded + follicle_snap_x/y; follicle_x/y are left untouched.
    # Runs after bridging so the detections it rescued are covered too.
    if follicle_snap:
        from .follicle_snap import add_follicle_snap
        out = add_follicle_snap(out, verbose=True)

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
            # Optional learned association cost. WW_ASSOC_MODEL points at a bundle
            # from association_model.py; without it the re-ranker behaves exactly
            # as before.
            _assoc = None
            _ap = os.environ.get("WW_ASSOC_MODEL")
            if _ap and not os.path.isabs(_ap):
                # bare filename resolves against models/, like WW_COVERAGE_MODEL --
                # an absolute host path is not necessarily visible inside a
                # container, which is exactly how the first A/B silently did nothing
                _ap = os.path.join(os.path.dirname(__file__), "models", _ap)
            if _ap and os.path.exists(_ap):
                try:
                    import joblib
                    _assoc = joblib.load(_ap)
                except Exception as exc:                       # noqa: BLE001
                    _stage(f"could not load association model {_ap}: {exc}")
            elif _ap:
                _stage(f"WW_ASSOC_MODEL={_ap!r} not found; ignoring")
            _wa = float(os.environ.get("WW_ASSOC_WEIGHT", "2.0"))
            out = _idm.rerank_identity(out, id_bundle, side_faces=side_faces,
                                       w_angle=w_angle, assoc_bundle=_assoc,
                                       w_assoc=_wa)
            _stage(f"identity re-ranker applied ({identity_mode}), "
                  f"w_angle={w_angle}"
                  + (f", learned association w={_wa}" if _assoc else "") + ".")

    out = out.sort_values(["fid", "wid"])
    output_path = output_path or combined_parquet.replace(".parquet", "_updated.parquet")
    out.to_parquet(output_path)
    _stage(f"saved {len(out)} rows to {output_path}")
    return output_path
