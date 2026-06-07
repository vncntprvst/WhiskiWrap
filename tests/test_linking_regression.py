"""End-to-end regression test of the linker on the example excerpt.

Runs link_whiskers on the raw tracking parquet and scores it against the
manually corrected ground truth, asserting the metrics stay above baseline.

Note on the ground truth: ``excerpt_video_updated_edited.parquet`` inherited a
cross-chunk identity split from the *old* tracker -- the left anterior whisker
is one continuous whisker but is labelled wid 2 outside frames 200-399 and wid 1
inside (the 200/400 boundaries are exactly WhiskiWrap's 200-frame chunk seams,
and the follicle position/length are continuous across them). The new linker
correctly keeps it as one identity, so we score against a corrected oracle that
merges left wid 1 into wid 2. See plan / eval_linking.py for details.
"""

import shutil

import pandas as pd
import pytest

from wwutils.classifiers import linker
from wwutils.classifiers import eval_linking as ev


def _corrected_gt(gt_path):
    gt = pd.read_parquet(gt_path)
    m = (gt["face_side"] == "left") & (gt["wid"] == 1)
    gt.loc[m, "wid"] = 2  # merge the chunk-split anterior whisker
    return gt


def test_linker_matches_corrected_groundtruth(tmp_path, example_raw_parquet,
                                               example_gt_parquet, example_whiskerpad):
    raw = tmp_path / "excerpt_video.parquet"
    shutil.copy(example_raw_parquet, raw)

    out = linker.link_whiskers(str(raw), example_whiskerpad)
    assert out is not None
    pred = pd.read_parquet(out)

    gt = _corrected_gt(example_gt_parquet)
    metrics = ev.compute_metrics(pred, gt)
    o = metrics["overall"]

    # Baseline measured: IDA/MOTA/IDF1 = 1.0, 0 switches, 0 spurious, worst MAE ~1.4.
    assert o["identity_accuracy"] >= 0.99, ev.format_report(metrics)
    assert o["idf1"] >= 0.99
    assert o["total_id_switches"] == 0
    assert o["extra_id_rows"] == 0, "spurious noise track survived linking"
    assert o["worst_angle_mae"] <= 3.0
    assert o["min_coverage"] >= 0.95


def test_linker_no_spurious_left_track(tmp_path, example_raw_parquet, example_whiskerpad):
    """The known low-score noise track in a corner must not become a whisker."""
    raw = tmp_path / "excerpt_video.parquet"
    shutil.copy(example_raw_parquet, raw)
    out = linker.link_whiskers(str(raw), example_whiskerpad)
    pred = pd.read_parquet(out)
    # Exactly 2 left + 3 right persistent whiskers expected on this clip.
    counts = pred.groupby("face_side")["wid"].nunique()
    assert counts["left"] == 2
    assert counts["right"] == 3
