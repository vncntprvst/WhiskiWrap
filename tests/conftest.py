"""Shared fixtures for the whisker-tracking test suite.

Provides small *synthetic* whisker dataframes (no video, no model, no
WhiskiWrap) so the linking tests are fast and self-contained, plus paths to the
example parquet files for the regression tests.

Synthetic dataframes carry both an unstable per-detection ``wid`` (what a tracer
emits) and a ``gt_wid`` column giving the true physical identity, so tests can
assert that linking recovers identity regardless of the raw ``wid``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# wwutils is installed (editable), so no sys.path manipulation is needed.
# Optional example data for the regression test lives in tests/data/ (skipped if absent).
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data")

# Per-detection defaults for columns the linker/eval don't drive but expect.
_DEFAULTS = dict(curvature=0.0, length=100.0, score=500.0, tip_x=0.0, tip_y=0.0,
                 face_x=0.0, face_y=0.0, chunk_start=0, pixel_length=50, label=0)


def make_detection(fid, wid, gt_wid, side, fx, fy, angle, **over):
    """Build a single detection row (dict)."""
    row = dict(fid=int(fid), wid=int(wid), gt_wid=int(gt_wid), face_side=side,
               follicle_x=float(fx), follicle_y=float(fy), angle=float(angle))
    row.update(_DEFAULTS)
    row.update(over)
    row.setdefault("pixels_x", np.array([fx, fx + 10.0]))
    row.setdefault("pixels_y", np.array([fy, fy + 10.0]))
    return row


def make_df(rows):
    """Build a whisker dataframe from a list of detection dicts."""
    return pd.DataFrame(rows)


@pytest.fixture
def detection_factory():
    return make_detection


@pytest.fixture
def df_factory():
    return make_df


# --------------------------------------------------------------------------- #
# Synthetic scenarios. Each returns a dataframe with gt_wid truth.
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth_clean():
    """Two well-separated whiskers, present every frame.

    Raw ``wid`` is deliberately swapped every other frame to prove the linker
    does not depend on it.
    """
    rows = []
    for fid in range(10):
        # whisker A (gt 0): top (small y), high angle. B (gt 1): bottom, low angle.
        order = [0, 1] if fid % 2 == 0 else [1, 0]  # unstable raw wid order
        rows.append(make_detection(fid, order[0], 0, "left", 100, 60 + 0.1 * fid, 65))
        rows.append(make_detection(fid, order[1], 1, "left", 100, 140 - 0.1 * fid, 35))
    return make_df(rows)


@pytest.fixture
def synth_crossing():
    """Two same-side whiskers whose follicle_y cross mid-clip.

    Positions coincide at the crossing, but angles stay distinct (A~65, B~35),
    so identity is recoverable via angle/velocity continuity though a naive
    per-frame position rank would swap them.
    """
    rows = []
    n = 11
    for fid in range(n):
        t = fid / (n - 1)
        ya = 60 + 80 * t        # A descends 60 -> 140
        yb = 140 - 80 * t       # B ascends 140 -> 60
        rows.append(make_detection(fid, 0, 0, "left", 100, ya, 65))
        rows.append(make_detection(fid, 1, 1, "left", 100, yb, 35))
    return make_df(rows)


@pytest.fixture
def synth_dropout():
    """Whisker B vanishes for frames 4-6 then returns (a brief gap)."""
    rows = []
    for fid in range(10):
        rows.append(make_detection(fid, 0, 0, "left", 100, 60, 65))
        if fid < 4 or fid > 6:
            rows.append(make_detection(fid, 1, 1, "left", 100, 140, 35))
    return make_df(rows)


@pytest.fixture
def synth_long_gap():
    """Whisker B vanishes for a 20-frame gap (longer than max_missed_frames).

    The forward pass closes B's track and opens a new one; Stage 2
    consolidation should re-stitch them since the endpoints are continuous.
    """
    rows = []
    for fid in range(40):
        rows.append(make_detection(fid, 0, 0, "left", 100, 60, 65))
        if fid < 10 or fid >= 30:
            rows.append(make_detection(fid, 1, 1, "left", 100, 140, 35))
    return make_df(rows)


@pytest.fixture
def synth_two_chunks():
    """Two clean whiskers spanning a chunk boundary at fid 5."""
    rows = []
    for fid in range(10):
        cs = 0 if fid < 5 else 5
        rows.append(make_detection(fid, 0, 0, "left", 100, 60, 65, chunk_start=cs))
        rows.append(make_detection(fid, 1, 1, "left", 100, 140, 35, chunk_start=cs))
    return make_df(rows)


# --------------------------------------------------------------------------- #
# Example-data paths (skip the test if missing).
# --------------------------------------------------------------------------- #
@pytest.fixture
def example_raw_parquet():
    p = os.path.join(EXAMPLE_DIR, "excerpt_video.parquet")
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    return p


@pytest.fixture
def example_gt_parquet():
    p = os.path.join(EXAMPLE_DIR, "excerpt_video_updated_edited.parquet")
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    return p


@pytest.fixture
def example_whiskerpad():
    p = os.path.join(EXAMPLE_DIR, "whiskerpad_excerpt_video.json")
    if not os.path.exists(p):
        pytest.skip(f"missing {p}")
    return p
