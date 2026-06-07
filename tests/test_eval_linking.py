"""Known-answer tests for the evaluation harness (eval_linking.py).

Build predicted/ground-truth dataframes whose metrics we can compute by hand,
and assert the harness reports them correctly.
"""

import numpy as np
import pandas as pd

from wwutils.classifiers import eval_linking as ev


def _gt(detection_factory):
    """Two stable left whiskers over 10 frames (gt wids 0 and 1)."""
    rows = []
    for fid in range(10):
        rows.append(detection_factory(fid, 0, 0, "left", 100, 60, 65))
        rows.append(detection_factory(fid, 1, 1, "left", 100, 140, 35))
    return pd.DataFrame(rows)


def test_perfect_match(detection_factory):
    gt = _gt(detection_factory)
    m = ev.compute_metrics(gt.copy(), gt.copy())
    o = m["overall"]
    assert o["identity_accuracy"] == 1.0
    assert o["mota"] == 1.0
    assert o["idf1"] == 1.0
    assert o["total_id_switches"] == 0
    assert o["extra_ids"] == []


def test_relabel_permutation_recovered(detection_factory):
    """A consistent relabel of wids must score perfectly (alignment recovers it)."""
    gt = _gt(detection_factory)
    pred = gt.copy()
    pred["wid"] = pred["wid"].map({0: 5, 1: 7})
    m = ev.compute_metrics(pred, gt)
    assert m["overall"]["identity_accuracy"] == 1.0
    assert m["wid_map"] == {5: 0, 7: 1}


def test_midway_swap_counts_id_switches(detection_factory):
    """Swapping the two ids at frame 5 yields exactly 2 id switches."""
    gt = _gt(detection_factory)
    pred = gt.copy()
    swap = pred["fid"] >= 5
    pred.loc[swap, "wid"] = pred.loc[swap, "wid"].map({0: 1, 1: 0})
    m = ev.compute_metrics(pred, gt)
    # One switch per gt whisker at the swap point.
    assert m["overall"]["total_id_switches"] == 2
    assert m["overall"]["mismatch"] > 0
    assert m["overall"]["identity_accuracy"] < 1.0


def test_spurious_track_flagged(detection_factory):
    """An extra predicted track not in GT is flagged without harming real IDA."""
    gt = _gt(detection_factory)
    pred = gt.copy()
    extra = [detection_factory(fid, 99, 99, "left", 400, 400, 120) for fid in range(10)]
    pred = pd.concat([pred, pd.DataFrame(extra)], ignore_index=True)
    m = ev.compute_metrics(pred, gt)
    assert 99 in m["overall"]["extra_ids"]
    assert m["overall"]["extra_id_rows"] == 10
    assert m["overall"]["identity_accuracy"] == 1.0  # real whiskers still perfect


def test_dropout_fragmentation(detection_factory):
    """A gap in the predicted track gives fragmentation == 2 for that whisker."""
    gt = _gt(detection_factory)
    pred = gt[~((gt["wid"] == 1) & (gt["fid"].between(4, 6)))].copy()
    m = ev.compute_metrics(pred, gt)
    assert m["per_gt_whisker"]["1"]["fragmentation"] == 2
    assert m["per_gt_whisker"]["1"]["coverage"] < 1.0
    assert m["per_gt_whisker"]["0"]["coverage"] == 1.0


def test_angle_mae_exact(detection_factory):
    """A constant +3 deg offset on one whisker yields angle_mae == 3.0."""
    gt = _gt(detection_factory)
    pred = gt.copy()
    pred.loc[pred["wid"] == 0, "angle"] += 3.0
    m = ev.compute_metrics(pred, gt)
    assert abs(m["per_gt_whisker"]["0"]["angle_mae"] - 3.0) < 1e-9
    assert m["per_gt_whisker"]["1"]["angle_mae"] < 1e-9


def test_angle_jump_counter(detection_factory):
    """An injected single-frame angle jump is counted."""
    gt = _gt(detection_factory)
    pred = gt.copy()
    pred.loc[(pred["wid"] == 0) & (pred["fid"] == 5), "angle"] += 50.0
    m = ev.compute_metrics(pred, gt, angle_jump_thresh=20.0)
    # Jump up at fid5 and back down at fid6 -> 2 consecutive-frame jumps.
    assert m["overall"]["pred_angle_jumps"] == 2
    assert m["overall"]["gt_angle_jumps"] == 0


def test_unequal_rows_false_positive(detection_factory):
    """An extra detection in one frame is counted as a false positive."""
    gt = _gt(detection_factory)
    pred = pd.concat(
        [gt.copy(), pd.DataFrame([detection_factory(0, 50, 50, "left", 250, 250, 100)])],
        ignore_index=True,
    )
    m = ev.compute_metrics(pred, gt)
    assert m["overall"]["false_positive"] == 1
