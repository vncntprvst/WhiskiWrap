"""Whisker linking: preserve whisker identity across frames and video chunks.

A behaviour video is traced per side and combined into a parquet with one row
per (frame, whisker detection). The raw detection ``wid`` is unstable: it is
re-assigned every frame and every WhiskiWrap chunk, so it cannot be used as a
persistent identity. This module assigns a *persistent* ``wid`` so that each id
refers to the same physical whisker for the whole video and per-whisker
kinematic traces (especially angle) are correct.

Design (see also eval_linking.py for how quality is measured):

* Each ``face_side`` is linked **independently** -- whiskers never cross between
  the left and right side of the face.
* A **single continuous forward pass** over the whole video links detections to
  tracks by optimal assignment (Hungarian) on a cost that combines follicle
  position (dominant), angle, curvature and length. Because the pass is
  continuous it never sees WhiskiWrap chunk seams, so identity is stable across
  chunks for free.
* Tracks survive **brief gaps** (occlusion / masking / a missed crossing) for up
  to ``max_missed_frames`` instead of spawning a new id.
* Identity is established by the *track*, never re-derived from per-frame
  spatial rank (which would swap ids at same-side crossings). Spatial order
  along the antero-posterior axis is used only **once** to assign stable,
  human-meaningful canonical ids (id 0 = most posterior).

Public entry point ``link_whiskers`` is a drop-in replacement for
``reclassify.reclassify``: same signature, same ``*_updated.parquet`` output and
return value. It depends only on pandas / numpy / scipy.

CLI:
    python linker.py INPUT.parquet WHISKERPAD.json
    python linker.py INPUT.parquet downward        # direction string also ok
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Maps ProtractionDirection -> (follicle axis column, ascending) such that
# canonical id 0 is the most posterior whisker. Mirrors the ordering convention
# previously implemented in reclassify.order_whiskers_per_frame.
_ORDER_RULES = {
    "downward":  ("follicle_y", False),  # bottom-most (largest y) first
    "upward":    ("follicle_y", True),   # top-most (smallest y) first
    "rightward": ("follicle_x", False),  # right-most (largest x) first
    "leftward":  ("follicle_x", True),   # left-most (smallest x) first
}


@dataclass
class LinkerParams:
    """Tunable parameters for the linker (defaults validated on the excerpt)."""
    # Stage 0 -- detection cleaning.
    length_frac: float = 0.25            # keep length > max(median_len)/ (1/frac)
    # Stage 1 -- forward assignment cost weights and gating.
    w_position: float = 1.0              # per pixel
    w_angle: float = 0.5                 # per degree
    w_curvature: float = 200.0           # per curvature unit (curvature is small)
    w_length: float = 0.1                # per length unit
    max_cost: float = 40.0               # reject assignments above this
    max_missed_frames: int = 15          # bridge gaps up to this many frames
    max_velocity_gap: int = 5            # cap velocity extrapolation (frames)
    # Stage 2 -- stitch tracklets across larger gaps the forward pass closed.
    # Off by default: the forward pass already bridges brief gaps via
    # max_missed_frames, and consolidation needs spatial noise filtering to be
    # safe on real data (otherwise scattered noise can be stitched into a track).
    # Enable once ground-truth on hard clips is available to tune it.
    consolidate: bool = False
    bridge_max_gap: int = 60             # max frame gap to bridge between tracklets
    bridge_max_dist: float = 25.0        # max follicle distance (px) across the gap
    bridge_max_angle: float = 25.0       # max angle difference (deg) across the gap
    # Stage 3 -- which tracks are real whiskers.
    min_frame_ratio: float = 0.3         # keep tracks present in >= this fraction
    # Output.
    add_corrected_angle: bool = True     # add angle_corrected (convention applied)


# --------------------------------------------------------------------------- #
# Stage 0: detection cleaning
# --------------------------------------------------------------------------- #
def clean_detections(df_side: pd.DataFrame, params: LinkerParams) -> pd.DataFrame:
    """Drop short fragment detections for one side.

    Threshold = max over raw whiskers of their median length, times
    ``length_frac``. This removes tiny spurious traces while keeping genuine
    (including short / partial) whiskers.
    """
    if df_side.empty:
        return df_side
    median_len = df_side.groupby("wid")["length"].median()
    threshold = median_len.max() * params.length_frac
    return df_side[df_side["length"] > threshold].copy()


# --------------------------------------------------------------------------- #
# Stage 1: continuous forward Hungarian linking
# --------------------------------------------------------------------------- #
def _assignment_cost(track: dict, det: pd.Series, fid: int, p: LinkerParams) -> float:
    """Cost of assigning ``det`` (this frame) to ``track`` (predicted state)."""
    gap = min(max(fid - track["last_fid"], 1), p.max_velocity_gap)
    pred_x = track["fx"] + track["vx"] * gap
    pred_y = track["fy"] + track["vy"] * gap
    pos = np.hypot(pred_x - det["follicle_x"], pred_y - det["follicle_y"])
    ang = abs(track["angle"] - det["angle"])
    cur = abs(track["curvature"] - det["curvature"])
    ln = abs(track["length"] - det["length"])
    return (p.w_position * pos + p.w_angle * ang
            + p.w_curvature * cur + p.w_length * ln)


def link_side_forward(df_side: pd.DataFrame, params: LinkerParams) -> pd.DataFrame:
    """Assign a persistent ``track_id`` to every detection of one side.

    Returns ``df_side`` with an added integer ``track_id`` column.
    """
    df_side = df_side.sort_values("fid").copy()
    track_ids = np.full(len(df_side), -1, dtype=int)
    pos_of_index = {idx: i for i, idx in enumerate(df_side.index)}

    tracks: List[dict] = []
    next_track_id = 0

    for fid, frame in df_side.groupby("fid", sort=True):
        dets = list(frame.itertuples(index=True))  # namedtuples; .Index is df index
        det_rows = [frame.loc[t.Index] for t in dets]

        active = [t for t in tracks if (fid - t["last_fid"]) <= params.max_missed_frames]

        if active and det_rows:
            cost = np.full((len(active), len(det_rows)), params.max_cost + 1.0)
            for i, tr in enumerate(active):
                for j, dr in enumerate(det_rows):
                    c = _assignment_cost(tr, dr, fid, params)
                    if c <= params.max_cost:
                        cost[i, j] = c
            rows, cols = linear_sum_assignment(cost)
            matched_tracks, matched_dets = set(), set()
            for r, c in zip(rows, cols):
                if cost[r, c] <= params.max_cost:
                    tr, dr = active[r], det_rows[c]
                    gap = max(fid - tr["last_fid"], 1)
                    tr["vx"] = (dr["follicle_x"] - tr["fx"]) / gap
                    tr["vy"] = (dr["follicle_y"] - tr["fy"]) / gap
                    tr.update(fx=dr["follicle_x"], fy=dr["follicle_y"],
                              angle=dr["angle"], curvature=dr["curvature"],
                              length=dr["length"], last_fid=fid)
                    track_ids[pos_of_index[dr.name]] = tr["track_id"]
                    matched_tracks.add(r)
                    matched_dets.add(c)
        else:
            matched_dets = set()

        # Unmatched detections -> new tracks.
        for j, dr in enumerate(det_rows):
            if j in matched_dets:
                continue
            tr = dict(track_id=next_track_id, fx=dr["follicle_x"], fy=dr["follicle_y"],
                      angle=dr["angle"], curvature=dr["curvature"], length=dr["length"],
                      vx=0.0, vy=0.0, last_fid=fid)
            tracks.append(tr)
            track_ids[pos_of_index[dr.name]] = next_track_id
            next_track_id += 1

    df_side["track_id"] = track_ids
    return df_side


# --------------------------------------------------------------------------- #
# Stage 2: tracklet consolidation across gaps
# --------------------------------------------------------------------------- #
def _track_endpoints(g: pd.DataFrame) -> dict:
    g = g.sort_values("fid")
    f, l = g.iloc[0], g.iloc[-1]
    return dict(first_fid=int(f["fid"]), last_fid=int(l["fid"]),
                sx=f["follicle_x"], sy=f["follicle_y"], sa=f["angle"],
                ex=l["follicle_x"], ey=l["follicle_y"], ea=l["angle"])


def consolidate_tracks(df_side: pd.DataFrame, params: LinkerParams) -> pd.DataFrame:
    """Merge non-overlapping tracklets that are the same whisker across a gap.

    The forward pass keeps a track alive only for ``max_missed_frames``; a longer
    occlusion closes it and starts a new track. This stage re-links such
    tracklets when one ends and another begins at a continuous follicle position
    and angle within ``bridge_max_gap`` frames. It is deliberately conservative:
    only temporally **non-overlapping** tracklets are ever merged, so two
    whiskers present at the same time are never combined.
    """
    if df_side.empty or df_side["track_id"].nunique() < 2:
        return df_side

    parent: Dict[int, int] = {t: t for t in df_side["track_id"].unique()}

    def rep(t: int) -> int:
        while parent[t] != t:
            t = parent[t]
        return t

    changed = True
    while changed:
        changed = False
        grouped = df_side.assign(_g=df_side["track_id"].map(rep)).groupby("_g")
        ginfo = {g: _track_endpoints(sub) for g, sub in grouped}
        best = None
        for a in ginfo:
            for b in ginfo:
                if a == b:
                    continue
                A, B = ginfo[a], ginfo[b]
                gap = B["first_fid"] - A["last_fid"]
                if gap < 1 or gap > params.bridge_max_gap:
                    continue
                dist = np.hypot(A["ex"] - B["sx"], A["ey"] - B["sy"])
                ang = abs(A["ea"] - B["sa"])
                if dist <= params.bridge_max_dist and ang <= params.bridge_max_angle:
                    cost = dist + ang
                    if best is None or cost < best[0]:
                        best = (cost, a, b)
        if best is not None:
            _, a, b = best
            parent[b] = a       # attach tracklet b after tracklet a
            changed = True

    df_side = df_side.copy()
    df_side["track_id"] = df_side["track_id"].map(rep)
    return df_side


# --------------------------------------------------------------------------- #
# Stage 3: canonical id assignment
# --------------------------------------------------------------------------- #
def assign_canonical_ids(df_side: pd.DataFrame, protraction: str,
                         params: LinkerParams, base_id: int) -> pd.DataFrame:
    """Keep frequent tracks and relabel them 0..n along the AP axis.

    ``base_id`` offsets ids so sides stay disjoint. Returns the filtered side
    with a ``canonical_wid`` column.
    """
    if df_side.empty:
        df_side["canonical_wid"] = []
        return df_side
    n_frames = df_side["fid"].nunique()
    counts = df_side.groupby("track_id")["fid"].nunique()
    keep = counts[counts >= params.min_frame_ratio * n_frames].index
    df_side = df_side[df_side["track_id"].isin(keep)].copy()
    if df_side.empty:
        df_side["canonical_wid"] = []
        return df_side

    axis, ascending = _ORDER_RULES.get(protraction, ("follicle_y", False))
    order = (df_side.groupby("track_id")[axis].mean()
             .sort_values(ascending=ascending).index.tolist())
    mapping = {tid: base_id + rank for rank, tid in enumerate(order)}
    df_side["canonical_wid"] = df_side["track_id"].map(mapping)
    return df_side


# --------------------------------------------------------------------------- #
# Stage 4: angle convention
# --------------------------------------------------------------------------- #
def apply_angle_convention(angles: pd.Series, image_side: Optional[str],
                           protraction: Optional[str]) -> pd.Series:
    """Apply the per-side angle convention.

    Negate when the face sits on the right of the cropped image, and mirror
    around 90 deg for a downward protraction. Mirrors
    BehaviorFun.angle_convention but reads the correct whiskerpad keys.
    """
    out = angles.copy()
    if image_side == "right":
        out = -out
    if protraction == "downward":
        out = 180 - out
    return out


# --------------------------------------------------------------------------- #
# whiskerpad parsing
# --------------------------------------------------------------------------- #
def _resolve_side_params(whiskerpad_arg: str) -> Dict[str, dict]:
    """Return {side: {'protraction':.., 'image_side':..}}.

    ``whiskerpad_arg`` may be a path to a whiskerpad JSON file or a bare
    protraction-direction string (in which case it applies to all sides and the
    image side is unknown, so the angle sign flip is skipped).
    """
    if os.path.isfile(whiskerpad_arg):
        with open(whiskerpad_arg, "r") as f:
            wp = json.load(f)
        out: Dict[str, dict] = {}
        for pad in wp["whiskerpads"]:
            out[pad["FaceSide"].lower()] = dict(
                protraction=pad.get("ProtractionDirection"),
                image_side=(pad.get("ImageSide") or "").lower() or None,
            )
        return out
    # Bare direction string: apply to any side, image side unknown.
    return {"*": dict(protraction=whiskerpad_arg, image_side=None)}


def _side_params(side_map: Dict[str, dict], side: str) -> dict:
    return side_map.get(side, side_map.get("*", dict(protraction=None, image_side=None)))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def link_whiskers(file_path: str, whiskerpad, plot: bool = False,
                  params: Optional[LinkerParams] = None) -> Optional[str]:
    """Link whiskers in a combined tracking parquet and save ``*_updated.parquet``.

    Parameters
    ----------
    file_path : str
        Combined per-side tracking parquet (output of combine_sides).
    whiskerpad : str
        Path to the whiskerpad JSON, or a bare protraction-direction string.
    plot : bool
        Unused placeholder kept for signature compatibility with reclassify.
    params : LinkerParams, optional
        Override default linking parameters.

    Returns
    -------
    str or None
        Path to the written ``*_updated.parquet`` file, or None on bad input.
    """
    if not str(file_path).endswith(".parquet"):
        print(f"File {file_path} is not a Parquet file. Exiting linking.")
        return None

    params = params or LinkerParams()
    df = pd.read_parquet(file_path)
    side_map = _resolve_side_params(whiskerpad)

    linked_sides: List[pd.DataFrame] = []
    base_id = 0
    for side in sorted(df["face_side"].unique()):
        sp = _side_params(side_map, side)
        df_side = df[df["face_side"] == side]
        cleaned = clean_detections(df_side, params)
        if cleaned.empty:
            print(f"[linker] {side}: no detections after cleaning, skipping.")
            continue
        linked = link_side_forward(cleaned, params)
        if params.consolidate:
            linked = consolidate_tracks(linked, params)
        canon = assign_canonical_ids(linked, sp["protraction"], params, base_id)
        if canon.empty:
            print(f"[linker] {side}: no persistent whiskers found.")
            continue
        n_whiskers = canon["canonical_wid"].nunique()
        base_id += n_whiskers
        print(f"[linker] {side}: {n_whiskers} persistent whiskers "
              f"(ids {sorted(canon['canonical_wid'].unique())}).")

        # Stage 4: finalise columns for this side. ``angle`` stays the raw
        # measurement (matches all existing data, overlays and notebooks); the
        # convention-corrected angle is added as a separate feature.
        if params.add_corrected_angle:
            canon["angle_corrected"] = apply_angle_convention(
                canon["angle"], sp["image_side"], sp["protraction"])
        canon["label"] = canon["wid"]                 # original detection id
        canon["wid"] = canon["canonical_wid"].astype(int)
        canon = canon.drop(columns=["canonical_wid"])
        linked_sides.append(canon)

    if not linked_sides:
        print("[linker] No whiskers linked on any side.")
        return None

    out_df = pd.concat(linked_sides, ignore_index=True).sort_values(["fid", "wid"])
    updated_path = file_path.replace(".parquet", "_updated.parquet")
    out_df.to_parquet(updated_path)
    print(f"[linker] Linking complete. Saved {len(out_df)} rows to {updated_path}")
    return updated_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Link whiskers across frames/chunks.")
    ap.add_argument("file_path", help="Combined tracking parquet")
    ap.add_argument("whiskerpad", help="Whiskerpad JSON path or protraction direction")
    ap.add_argument("--plot", action="store_true", help="(reserved; no-op)")
    args = ap.parse_args()
    link_whiskers(args.file_path, args.whiskerpad, args.plot)


if __name__ == "__main__":
    main()
