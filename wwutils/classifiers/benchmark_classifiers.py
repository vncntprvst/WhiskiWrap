#!/usr/bin/env python3
"""Benchmark whisker ID classifiers.

This script compares the performance of the available whisker
classifiers on a misclassified tracking file. The results are compared
against a reference file with correct whisker IDs and reported as
accuracy and runtime.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import pandas as pd

from . import reclassify as rc
from .unet_classifier import assign_whisker_ids
from .gnn_classifier import reassign_whisker_ids_gnn


def _accuracy(pred: pd.DataFrame, true: pd.DataFrame) -> float:
    """Return fraction of matching whisker IDs."""
    if len(pred) != len(true):
        raise ValueError("Prediction and reference must have same length")
    return (pred["wid"].values == true["wid"].values).mean()


def benchmark(
    misclassified: str,
    reference: str,
    video: str | None = None,
    whiskerpad: str | None = None,
    gnn_model: str | None = None,
    gnn_epochs: int = 50,
) -> List[Tuple[str, float, float]]:
    """Run each classifier and report (name, time, accuracy)."""
    bad_df = pd.read_parquet(misclassified)
    ref_df = pd.read_parquet(reference)
    results: List[Tuple[str, float, float]] = []

    if video:
        start = time.time()
        out = assign_whisker_ids(video, misclassified, model_path=None)
        runtime = time.time() - start
        if out and os.path.exists(out):
            unet_df = pd.read_parquet(out)
            acc = _accuracy(unet_df, ref_df)
            results.append(("unet", runtime, acc))
            try:
                os.remove(out)
            except OSError:
                pass

    start = time.time()
    rc_out = rc.reclassify(misclassified, whiskerpad or "downward", plot=False)
    rc_runtime = time.time() - start
    if rc_out and os.path.exists(rc_out):
        rc_df = pd.read_parquet(rc_out)
        acc = _accuracy(rc_df, ref_df)
        results.append(("reclassify", rc_runtime, acc))
        try:
            os.remove(rc_out)
        except OSError:
            pass

    start = time.time()
    gnn_df, _ = reassign_whisker_ids_gnn(
        bad_df,
        model_path=gnn_model,
        n_epochs=gnn_epochs,
        temporal_window=10,
        train_split=0.8,
        verbose=False,
    )
    gnn_runtime = time.time() - start
    acc = _accuracy(gnn_df, ref_df)
    results.append(("gnn", gnn_runtime, acc))

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark whisker ID classifiers")
    p.add_argument("misclassified", help="Parquet file with misclassified IDs")
    p.add_argument("reference", help="Reference parquet with correct IDs")
    p.add_argument("--video", help="Video file for U-Net classifier")
    p.add_argument("--whiskerpad", help="Whiskerpad JSON or direction for reclassify")
    p.add_argument("--gnn-model", help="Path to pretrained GNN model")
    p.add_argument("--epochs", type=int, default=50, help="Training epochs for GNN")
    args = p.parse_args()

    results = benchmark(
        args.misclassified,
        args.reference,
        video=args.video,
        whiskerpad=args.whiskerpad,
        gnn_model=args.gnn_model,
        gnn_epochs=args.epochs,
    )

    print("\nClassifier benchmark results:")
    for name, runtime, acc in results:
        print(f"{name:12s} accuracy={acc:.3f} runtime={runtime:.2f}s")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

