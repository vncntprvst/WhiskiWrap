"""Whisker ID prediction using a Graph Neural Network.

This module provides utilities to apply a GNN model to whisker tracking
parquet files. If no pretrained model is supplied the classifier can train a
model on the provided data before predicting new whisker IDs.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np

# Optional dependencies -----------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False

try:
    import torch_geometric  # noqa: F401
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Feature extraction and graph building
# ---------------------------------------------------------------------------

def extract_whisker_features(whisker_data: pd.DataFrame) -> np.ndarray:
    """Return feature matrix for a whisker dataframe."""
    feature_columns = ["angle", "curvature", "length", "follicle_x", "follicle_y"]
    if "tip_x" in whisker_data.columns and "tip_y" in whisker_data.columns:
        feature_columns += ["tip_x", "tip_y"]

    if len(whisker_data["fid"].unique()) > 1:
        whisker_data = whisker_data.sort_values(["wid", "fid"])
        for col in ["angle", "follicle_x", "follicle_y"]:
            if col in whisker_data.columns:
                whisker_data[f"{col}_velocity"] = whisker_data.groupby("wid")[col].diff()
                feature_columns.append(f"{col}_velocity")

    available = [c for c in feature_columns if c in whisker_data.columns]
    if not available:
        warnings.warn("No whisker features found. Using random fallback features.")
        return np.random.randn(len(whisker_data), 5)

    feats = whisker_data[available].values
    return np.nan_to_num(feats, nan=0.0)


def build_whisker_graph(
    whisker_data: pd.DataFrame, temporal_window: int = 10
) -> Tuple[List[Data], Dict[int, int], Dict[int, int]]:
    """Create torch-geometric graphs representing temporal whisker relations."""
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required for graph construction")

    unique_wids = sorted(whisker_data["wid"].unique())
    id_to_class = {wid: i for i, wid in enumerate(unique_wids)}
    class_to_id = {i: wid for i, wid in enumerate(unique_wids)}

    graphs: List[Data] = []
    frames = sorted(whisker_data["fid"].unique())

    for i, frame in enumerate(frames):
        start = max(0, i - temporal_window // 2)
        end = min(len(frames), i + temporal_window // 2 + 1)
        context_frames = frames[start:end]
        context = whisker_data[whisker_data["fid"].isin(context_frames)]
        if context.empty:
            continue

        x = torch.tensor(extract_whisker_features(context), dtype=torch.float)
        edge_indices: List[List[int]] = []

        for cf in context_frames:
            fdata = context[context["fid"] == cf]
            idxs = fdata.index.tolist()
            for j in range(len(idxs)):
                for k in range(j + 1, len(idxs)):
                    g1 = context.index.get_loc(idxs[j])
                    g2 = context.index.get_loc(idxs[k])
                    edge_indices.append([g1, g2])
                    edge_indices.append([g2, g1])

        for cf_idx in range(len(context_frames) - 1):
            cf = context_frames[cf_idx]
            nf = context_frames[cf_idx + 1]
            cdata = context[context["fid"] == cf]
            ndata = context[context["fid"] == nf]
            for c_idx in cdata.index:
                for n_idx in ndata.index:
                    g_c = context.index.get_loc(c_idx)
                    g_n = context.index.get_loc(n_idx)
                    edge_indices.append([g_c, g_n])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        y = torch.tensor([id_to_class[w] for w in context["wid"].values], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs, id_to_class, class_to_id


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

if TORCH_GEOMETRIC_AVAILABLE:

    class WhiskerGCN(nn.Module):
        """Graph convolution network for whisker ID prediction."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 10):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x, edge_index):  # pragma: no cover - simple forward
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)

else:  # pragma: no cover

    class WhiskerGCN:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch_geometric is required for WhiskerGCN")


if TORCH_AVAILABLE:

    class GNNWhiskerTracker:
        """Trainable tracker using a GNN model."""

        def __init__(self, n_whiskers: int = 5, temporal_window: int = 10, device: str = "auto"):
            self.n_whiskers = n_whiskers
            self.temporal_window = temporal_window
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            self.model: Optional[nn.Module] = None
            self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
            self.input_dim: Optional[int] = None
            self.id_to_class: Dict[int, int] = {}
            self.class_to_id: Dict[int, int] = {}

        def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
            feats = extract_whisker_features(df)
            if self.scaler is not None:
                if hasattr(self.scaler, "scale_"):
                    feats = self.scaler.transform(feats)
                else:
                    feats = self.scaler.fit_transform(feats)
            self.input_dim = feats.shape[1]
            return df

        def train(
            self,
            whisker_data: pd.DataFrame,
            n_epochs: int = 100,
            train_split: float = 0.8,
            learning_rate: float = 0.01,
            verbose: bool = True,
        ) -> Dict[str, List[float]]:
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise ImportError("torch_geometric is required for training")

            processed = self._prepare_features(whisker_data)
            graphs, id_to_class, class_to_id = build_whisker_graph(processed, self.temporal_window)
            self.id_to_class = id_to_class
            self.class_to_id = class_to_id
            if not graphs:
                raise ValueError("No graphs could be built from the data")

            if len(graphs) > 1:
                train_graphs, val_graphs = train_test_split(graphs, train_size=train_split, random_state=42)
            else:
                train_graphs, val_graphs = graphs, []

            self.model = WhiskerGCN(self.input_dim, 64, self.n_whiskers).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.NLLLoss()
            train_losses: List[float] = []
            val_losses: List[float] = []

            for epoch in range(n_epochs):
                self.model.train()
                t_loss = 0.0
                for g in train_graphs:
                    g = g.to(self.device)
                    optimizer.zero_grad()
                    out = self.model(g.x, g.edge_index)
                    loss = criterion(out, g.y)
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()
                t_loss /= len(train_graphs)
                train_losses.append(t_loss)

                if val_graphs:
                    self.model.eval()
                    v_loss = 0.0
                    with torch.no_grad():
                        for g in val_graphs:
                            g = g.to(self.device)
                            out = self.model(g.x, g.edge_index)
                            loss = criterion(out, g.y)
                            v_loss += loss.item()
                    v_loss /= len(val_graphs)
                    val_losses.append(v_loss)
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss={t_loss:.4f}, Val Loss={v_loss:.4f}")
                else:
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss={t_loss:.4f}")

            history = {"train_loss": train_losses}
            if val_losses:
                history["val_loss"] = val_losses
            return history

        def predict(self, whisker_data: pd.DataFrame) -> np.ndarray:
            if self.model is None:
                raise ValueError("Model not trained or loaded")

            processed = self._prepare_features(whisker_data)
            graphs, _, _ = build_whisker_graph(processed, self.temporal_window)
            if not graphs:
                return whisker_data["wid"].values

            frames = sorted(whisker_data["fid"].unique())
            frame_predictions: Dict[int, List[int]] = {}
            self.model.eval()
            with torch.no_grad():
                for i, g in enumerate(graphs):
                    if i >= len(frames):
                        break
                    g = g.to(self.device)
                    out = self.model(g.x, g.edge_index)
                    preds = torch.argmax(out, dim=1)
                    target_frame = frames[i]
                    frame_data = whisker_data[whisker_data["fid"] == target_frame]
                    if len(preds) >= len(frame_data):
                        pred_slice = preds[: len(frame_data)]
                        mapped = [self.class_to_id[p.item()] for p in pred_slice]
                        frame_predictions[target_frame] = mapped
                    else:
                        frame_predictions[target_frame] = frame_data["wid"].tolist()

            results: List[int] = []
            for _, row in whisker_data.iterrows():
                fid = row["fid"]
                if fid in frame_predictions:
                    idx = whisker_data[whisker_data["fid"] == fid].index.get_loc(row.name)
                    if idx < len(frame_predictions[fid]):
                        results.append(frame_predictions[fid][idx])
                    else:
                        results.append(row["wid"])
                else:
                    results.append(row["wid"])
            return np.array(results)

        def save_model(self, path: str) -> None:
            if self.model is None:
                raise ValueError("No model to save")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "scaler": self.scaler,
                    "n_whiskers": self.n_whiskers,
                    "temporal_window": self.temporal_window,
                    "input_dim": self.input_dim,
                    "id_to_class": self.id_to_class,
                    "class_to_id": self.class_to_id,
                },
                path,
            )

        def load_model(self, path: str) -> None:
            checkpoint = torch.load(path, map_location=self.device)
            self.n_whiskers = checkpoint["n_whiskers"]
            self.temporal_window = checkpoint["temporal_window"]
            self.input_dim = checkpoint["input_dim"]
            self.scaler = checkpoint["scaler"]
            self.id_to_class = checkpoint.get("id_to_class", {})
            self.class_to_id = checkpoint.get("class_to_id", {})
            self.model = WhiskerGCN(self.input_dim, 64, self.n_whiskers).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

else:  # pragma: no cover

    class GNNWhiskerTracker:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for GNNWhiskerTracker")


# ---------------------------------------------------------------------------
# High level API
# ---------------------------------------------------------------------------

def _fallback_reassignment(
    whisker_data: pd.DataFrame, temporal_window: int = 10, verbose: bool = True
) -> pd.DataFrame:
    if verbose:
        print("Using fallback ID reassignment based on spatial consistency...")

    result = whisker_data.copy()
    for side in result["face_side"].unique():
        mask = result["face_side"] == side
        frames = sorted(result.loc[mask, "fid"].unique())
        for i in range(1, len(frames)):
            prev_f = frames[i - 1]
            curr_f = frames[i]
            mask_prev = mask & (result["fid"] == prev_f)
            mask_curr = mask & (result["fid"] == curr_f)
            prev_rows = result.loc[mask_prev]
            curr_rows = result.loc[mask_curr]
            if prev_rows.empty or curr_rows.empty:
                continue
            prev_pos = prev_rows[["follicle_x", "follicle_y"]].values
            curr_pos = curr_rows[["follicle_x", "follicle_y"]].values
            dists = np.linalg.norm(curr_pos[:, None, :] - prev_pos[None, :, :], axis=2)
            used = set()
            new_ids = []
            for j in range(dists.shape[0]):
                choices = [(d, idx) for idx, d in enumerate(dists[j]) if idx not in used]
                if choices:
                    _, best = min(choices)
                    used.add(best)
                    new_ids.append(prev_rows.iloc[best]["wid"])
                else:
                    new_ids.append(curr_rows.iloc[j]["wid"])
            for idx, wid in zip(curr_rows.index, new_ids):
                result.at[idx, "wid"] = wid
    if verbose:
        print("Fallback ID reassignment completed.")
    return result


def reassign_whisker_ids_gnn(
    whisker_data: pd.DataFrame,
    model_path: Optional[str] = None,
    n_epochs: int = 100,
    temporal_window: int = 10,
    train_split: float = 0.8,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, bool]:
    """Reassign whisker IDs using a GNN model."""
    if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        warnings.warn(
            "PyTorch and/or torch_geometric not available. Using fallback method.")
        return _fallback_reassignment(whisker_data, temporal_window, verbose), False

    if not SKLEARN_AVAILABLE:
        warnings.warn("sklearn not available. Using basic preprocessing.")

    try:
        if verbose:
            print("Initializing GNN-based whisker ID reassignment...")
        tracker = GNNWhiskerTracker(
            n_whiskers=whisker_data["wid"].nunique(),
            temporal_window=temporal_window,
        )
        if model_path is None or not os.path.exists(model_path):
            if verbose:
                print("Training new GNN model...")
            tracker.train(whisker_data, n_epochs=n_epochs, train_split=train_split, verbose=verbose)
        else:
            if verbose:
                print(f"Loading pretrained model from {model_path}")
            tracker.load_model(model_path)
        new_ids = tracker.predict(whisker_data)
        result = whisker_data.copy()
        result["wid"] = new_ids
        if verbose:
            diff = (whisker_data["wid"] != result["wid"]).sum()
            print(f"GNN ID reassignment completed. Changed {diff} IDs.")
        return result, True
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"GNN reassignment failed: {exc}. Using fallback method.")
        return _fallback_reassignment(whisker_data, temporal_window, verbose), False


def assign_whisker_ids(
    parquet_file: str,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None,
    n_epochs: int = 100,
    train_split: float = 0.8,
    temporal_window: int = 10,
) -> Optional[str]:
    """Assign whisker IDs on a parquet file and save the result."""
    df = pd.read_parquet(parquet_file)
    result, _ = reassign_whisker_ids_gnn(
        df,
        model_path=model_path,
        n_epochs=n_epochs,
        temporal_window=temporal_window,
        train_split=train_split,
        verbose=True,
    )
    out_file = output_file or parquet_file.replace(".parquet", "_gnn.parquet")
    result.to_parquet(out_file, index=False)
    return out_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Assign whisker IDs using a GNN model")
    p.add_argument("parquet_file", help="Path to tracking parquet file")
    p.add_argument("--model", help="Path to a trained GNN model")
    p.add_argument("--output", help="Output parquet file")
    p.add_argument("--epochs", type=int, default=100, help="Training epochs if no model is provided")
    p.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    p.add_argument("--temporal-window", type=int, default=10, help="Temporal window size")
    args = p.parse_args()

    out = assign_whisker_ids(
        args.parquet_file,
        model_path=args.model,
        output_file=args.output,
        n_epochs=args.epochs,
        train_split=args.train_split,
        temporal_window=args.temporal_window,
    )
    print(f"Results saved to: {out}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
