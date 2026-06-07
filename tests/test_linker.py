"""Unit tests for the whisker linker (linker.py).

These use small synthetic dataframes (see conftest.py) that exercise the cases
the linker must handle: clean separation, same-side crossing, brief dropout, and
chunk boundaries. Identity correctness is checked by requiring that each linked
track contains exactly one ground-truth whisker (no id mixing) and that each
ground-truth whisker is followed by a single track.
"""

import numpy as np
import pandas as pd

from wwutils.classifiers import linker


def _assert_no_id_mixing(linked):
    """Each track_id must map to exactly one ground-truth whisker."""
    mixing = linked.groupby("track_id")["gt_wid"].nunique()
    assert (mixing == 1).all(), f"track(s) mixed identities: {mixing[mixing > 1].to_dict()}"


def _tracks_per_gt(linked):
    return linked.groupby("gt_wid")["track_id"].nunique()


def test_clean_two_tracks(synth_clean):
    linked = linker.link_side_forward(synth_clean, linker.LinkerParams())
    _assert_no_id_mixing(linked)
    # Exactly two whiskers, each a single continuous track.
    assert (_tracks_per_gt(linked) == 1).all()
    assert linked["track_id"].nunique() == 2


def test_crossing_preserves_identity(synth_crossing):
    """Identity must survive a same-side crossing (no swap)."""
    linked = linker.link_side_forward(synth_crossing, linker.LinkerParams())
    _assert_no_id_mixing(linked)
    assert (_tracks_per_gt(linked) == 1).all(), "identity swapped at the crossing"


def test_dropout_bridged(synth_dropout):
    """A brief gap (3 frames) must be bridged into one track."""
    linked = linker.link_side_forward(synth_dropout, linker.LinkerParams(max_missed_frames=15))
    _assert_no_id_mixing(linked)
    # whisker B (gt 1) reappears after the gap under the same track.
    assert _tracks_per_gt(linked)[1] == 1


def test_dropout_not_bridged_when_gap_exceeds_limit(synth_dropout):
    """If the gap exceeds max_missed_frames, a new track is started."""
    linked = linker.link_side_forward(synth_dropout, linker.LinkerParams(max_missed_frames=2))
    _assert_no_id_mixing(linked)  # still no mixing, just fragmented
    assert _tracks_per_gt(linked)[1] == 2


def test_consolidate_bridges_long_gap(synth_long_gap):
    """Stage 2 re-stitches tracklets separated by a gap > max_missed_frames."""
    params = linker.LinkerParams(max_missed_frames=15)
    linked = linker.link_side_forward(synth_long_gap, params)
    # Forward pass alone leaves B fragmented across the 20-frame gap.
    assert _tracks_per_gt(linked)[1] == 2
    consolidated = linker.consolidate_tracks(linked, params)
    _assert_no_id_mixing(consolidated)
    assert _tracks_per_gt(consolidated)[1] == 1, "long gap was not bridged"


def test_consolidate_never_merges_overlapping(synth_clean):
    """Two simultaneously-present whiskers must never be merged."""
    params = linker.LinkerParams()
    linked = linker.link_side_forward(synth_clean, params)
    consolidated = linker.consolidate_tracks(linked, params)
    assert consolidated["track_id"].nunique() == 2


def test_keep_top_n_per_frame_drops_noise(detection_factory, df_factory):
    """top-N-by-length keeps the long real whiskers and drops short noise."""
    rows = []
    for fid in range(5):
        rows.append(detection_factory(fid, 0, 0, "left", 100, 60, 65, length=200))
        rows.append(detection_factory(fid, 1, 1, "left", 100, 140, 35, length=190))
        rows.append(detection_factory(fid, 9, 9, "left", 300, 300, 110, length=20))  # noise
    kept = linker.keep_top_n_per_frame(df_factory(rows), 2)
    assert set(kept["gt_wid"]) == {0, 1}
    assert (kept.groupby("fid").size() == 2).all()


def test_two_chunks_stable_identity(synth_two_chunks):
    """Identity is stable across a chunk boundary (linker ignores chunk_start)."""
    linked = linker.link_side_forward(synth_two_chunks, linker.LinkerParams())
    _assert_no_id_mixing(linked)
    assert (_tracks_per_gt(linked) == 1).all()


def test_clean_detections_drops_short_fragments(detection_factory, df_factory):
    """Stage 0 removes short fragments below the length threshold."""
    rows = []
    for fid in range(10):
        rows.append(detection_factory(fid, 0, 0, "left", 100, 60, 65, length=120))
        rows.append(detection_factory(fid, 9, 9, "left", 300, 300, 110, length=10))  # noise
    df = df_factory(rows)
    cleaned = linker.clean_detections(df, linker.LinkerParams())
    assert set(cleaned["gt_wid"]) == {0}


def test_canonical_ids_ordered_posterior_first(synth_clean):
    """Downward protraction -> id 0 is the bottom-most (largest follicle_y)."""
    linked = linker.link_side_forward(synth_clean, linker.LinkerParams())
    canon = linker.assign_canonical_ids(linked, "downward", linker.LinkerParams(), base_id=0)
    # gt 1 sits at larger y (140) than gt 0 (60), so gt 1 should be canonical 0.
    id_for_gt = canon.groupby("gt_wid")["canonical_wid"].first()
    assert id_for_gt[1] < id_for_gt[0]


def test_link_whiskers_end_to_end(tmp_path, synth_clean):
    """Full link_whiskers writes *_updated.parquet with expected schema/ids."""
    src = tmp_path / "synthetic.parquet"
    # link_whiskers reads/writes parquet; drop the helper gt column first.
    synth_clean.drop(columns=["gt_wid"]).to_parquet(src)
    out = linker.link_whiskers(str(src), "downward")
    assert out is not None and out.endswith("_updated.parquet")
    res = pd.read_parquet(out)
    # Two persistent whiskers, each one stable wid across all frames.
    assert res["wid"].nunique() == 2
    per_frame_ids = res.groupby("fid")["wid"].apply(lambda s: tuple(sorted(s)))
    assert per_frame_ids.nunique() == 1  # same id set every frame
    for col in ("wid", "label", "angle", "angle_corrected", "track_id", "face_side"):
        assert col in res.columns
