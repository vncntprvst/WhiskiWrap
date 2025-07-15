"""Train a GNN model for whisker ID reassignment.

Inputs:
    * Parquet file with whisker tracking data
    * --output: Output path for the trained model
    * --epochs: Number of training epochs
    * --train-split: Training split ratio
    * --temporal-window: Temporal window size

Usage:
    python train_gnn.py <parquet_file> [--output <model_output_path>]

"""

from __future__ import annotations

import os
import argparse
import pandas as pd

# Try absolute import first, fall back to relative if running as module
try:
    from wwutils.classifiers.gnn_classifier import GNNWhiskerTracker
except ImportError:
    try:
        from .gnn_classifier import GNNWhiskerTracker
    except ImportError:
        print("Error: Could not import GNNWhiskerTracker. Make sure you're running from the project root.")
        exit(1)
        
def train(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.parquet)
    tracker = GNNWhiskerTracker(n_whiskers=df["wid"].nunique(), temporal_window=args.temporal_window)
    tracker.train(df, n_epochs=args.epochs, train_split=args.train_split, verbose=True)
    output_path = args.output or os.path.join(os.path.dirname(__file__), "gnn_model.pt")
    tracker.save_model(output_path)
    print(f"Model saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GNN whisker classifier")
    p.add_argument("parquet", help="Tracking parquet file for training")
    p.add_argument("--output", help="Path to save trained model")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    p.add_argument("--temporal-window", type=int, default=10, help="Temporal window size")
    return p.parse_args()


if __name__ == "__main__":  # pragma: no cover - CLI
    train(parse_args())
