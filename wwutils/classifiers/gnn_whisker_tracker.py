"""
GNN-based whisker ID reassignment for temporal consistency.

This module provides a Graph Neural Network (GNN) approach for reassigning whisker IDs
to maintain temporal consistency across video frames, inspired by PoseGCN.
"""

import os
import warnings
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import argparse

# Check for optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def extract_whisker_features(whisker_data: pd.DataFrame) -> np.ndarray:
    """
    Extract features from whisker tracking data.
    
    Args:
        whisker_data: DataFrame with whisker measurements
        
    Returns:
        Feature matrix (n_whiskers, n_features)
    """
    features = []
    
    # Basic geometric features
    feature_columns = ['angle', 'curvature', 'length', 'follicle_x', 'follicle_y']
    
    # Add tip position if available
    if 'tip_x' in whisker_data.columns and 'tip_y' in whisker_data.columns:
        feature_columns.extend(['tip_x', 'tip_y'])
    
    # Add velocity features if multiple frames available
    if len(whisker_data['fid'].unique()) > 1:
        # Calculate velocity features
        whisker_data = whisker_data.sort_values(['wid', 'fid'])
        
        for col in ['angle', 'follicle_x', 'follicle_y']:
            if col in whisker_data.columns:
                whisker_data[f'{col}_velocity'] = whisker_data.groupby('wid')[col].diff()
                feature_columns.append(f'{col}_velocity')
    
    # Select features that exist in the data
    available_features = [col for col in feature_columns if col in whisker_data.columns]
    
    if not available_features:
        # Fallback to basic features
        warnings.warn("No standard whisker features found. Using basic fallback features.")
        features_matrix = np.random.randn(len(whisker_data), 5)  # Random features as fallback
    else:
        features_matrix = whisker_data[available_features].values
        
        # Handle NaN values
        features_matrix = np.nan_to_num(features_matrix, nan=0.0)
    
    return features_matrix


def build_whisker_graph(whisker_data: pd.DataFrame, 
                       temporal_window: int = 10) -> Tuple[List, Dict[int, int], Dict[int, int]]:
    """
    Build graph data for whisker tracking.
    
    Args:
        whisker_data: DataFrame with whisker measurements
        temporal_window: Temporal context window
        
    Returns:
        Tuple of (List of PyTorch Geometric Data objects, id_to_class mapping, class_to_id mapping)
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required for graph construction")
    
    # Create mapping from whisker IDs to 0-based class indices
    unique_wids = sorted(whisker_data['wid'].unique())
    id_to_class = {wid: i for i, wid in enumerate(unique_wids)}
    class_to_id = {i: wid for i, wid in enumerate(unique_wids)}
    
    graphs = []
    frames = sorted(whisker_data['fid'].unique())
    
    for i, frame in enumerate(frames):
        # Get temporal context window
        start_frame = max(0, i - temporal_window // 2)
        end_frame = min(len(frames), i + temporal_window // 2 + 1)
        
        context_frames = frames[start_frame:end_frame]
        context_data = whisker_data[whisker_data['fid'].isin(context_frames)]
        
        if len(context_data) == 0:
            continue
            
        # Extract features
        features = extract_whisker_features(context_data)
        
        # Create node features
        x = torch.tensor(features, dtype=torch.float)
        
        # Create edges (spatial and temporal)
        edge_indices = []
        
        # Add spatial edges within each frame
        for cf in context_frames:
            frame_data = context_data[context_data['fid'] == cf]
            frame_indices = frame_data.index.tolist()
            
            # Connect all whiskers in the same frame
            for idx1 in range(len(frame_indices)):
                for idx2 in range(idx1 + 1, len(frame_indices)):
                    global_idx1 = context_data.index.get_loc(frame_indices[idx1])
                    global_idx2 = context_data.index.get_loc(frame_indices[idx2])
                    edge_indices.append([global_idx1, global_idx2])
                    edge_indices.append([global_idx2, global_idx1])  # Undirected
        
        # Add temporal edges between consecutive frames
        for cf_idx in range(len(context_frames) - 1):
            curr_frame = context_frames[cf_idx]
            next_frame = context_frames[cf_idx + 1]
            
            curr_data = context_data[context_data['fid'] == curr_frame]
            next_data = context_data[context_data['fid'] == next_frame]
            
            # Connect whiskers across frames (fully connected for simplicity)
            for curr_idx in curr_data.index:
                for next_idx in next_data.index:
                    global_curr = context_data.index.get_loc(curr_idx)
                    global_next = context_data.index.get_loc(next_idx)
                    edge_indices.append([global_curr, global_next])
        
        # Create edge index tensor
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create labels (whisker IDs mapped to 0-based classes)
        y = torch.tensor([id_to_class[wid] for wid in context_data['wid'].values], dtype=torch.long)
        
        # Create graph
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)
    
    return graphs, id_to_class, class_to_id


if TORCH_GEOMETRIC_AVAILABLE:
    class WhiskerGCN(nn.Module):
        """
        Graph Convolutional Network for whisker ID prediction.
        """
        
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 10):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, num_classes)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x, edge_index, batch=None):
            # First GCN layer
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Second GCN layer
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Output layer
            x = self.conv3(x, edge_index)
            
            return F.log_softmax(x, dim=1)
else:
    # Dummy class when torch_geometric is not available
    class WhiskerGCN:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch_geometric is required for WhiskerGCN")


if TORCH_AVAILABLE:
    class GNNWhiskerTracker:
        """
        GNN-based whisker tracker for ID reassignment.
        """
        
        def __init__(self, n_whiskers: int = 5, temporal_window: int = 10, 
                     device: str = 'auto'):
            """
            Initialize GNN whisker tracker.
            
            Args:
                n_whiskers: Number of whiskers to track
                temporal_window: Temporal context window
                device: Device to use ('cuda', 'cpu', or 'auto')
            """
            self.n_whiskers = n_whiskers
            self.temporal_window = temporal_window
            
            if device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            
            self.model = None
            self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
            self.input_dim = None
            
        def _prepare_features(self, whisker_data: pd.DataFrame) -> pd.DataFrame:
            """Prepare and normalize features."""
            processed_data = whisker_data.copy()
            
            # Extract features
            features = extract_whisker_features(processed_data)
            
            # Normalize features if scaler is available
            if self.scaler is not None:
                if hasattr(self.scaler, 'scale_'):
                    features = self.scaler.transform(features)
                else:
                    features = self.scaler.fit_transform(features)
            
            # Update input dimension
            self.input_dim = features.shape[1]
            
            return processed_data
        
        def train(self, whisker_data: pd.DataFrame, n_epochs: int = 100, 
                  train_split: float = 0.8, learning_rate: float = 0.01,
                  verbose: bool = True) -> Dict[str, List[float]]:
            """
            Train the GNN model.
            
            Args:
                whisker_data: Training data
                n_epochs: Number of training epochs
                train_split: Training split ratio
                learning_rate: Learning rate
                verbose: Whether to print progress
                
            Returns:
                Training history
            """
            if not TORCH_GEOMETRIC_AVAILABLE:
                raise ImportError("torch_geometric is required for training")
            
            # Prepare features
            processed_data = self._prepare_features(whisker_data)
            
            # Build graphs
            graphs, id_to_class, class_to_id = build_whisker_graph(processed_data, self.temporal_window)
            
            # Store the mappings
            self.id_to_class = id_to_class
            self.class_to_id = class_to_id
            
            if len(graphs) == 0:
                raise ValueError("No graphs could be built from the data")
            
            # Split data
            if len(graphs) > 1:
                train_graphs, val_graphs = train_test_split(graphs, train_size=train_split, 
                                                          random_state=42)
            else:
                train_graphs = graphs
                val_graphs = []
            
            # Initialize model
            self.model = WhiskerGCN(
                input_dim=self.input_dim,
                hidden_dim=64,
                num_classes=self.n_whiskers
            ).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.NLLLoss()
            
            # Training loop
            train_losses = []
            val_losses = []
            
            for epoch in range(n_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for graph in train_graphs:
                    graph = graph.to(self.device)
                    
                    optimizer.zero_grad()
                    out = self.model(graph.x, graph.edge_index)
                    loss = criterion(out, graph.y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_graphs)
                train_losses.append(train_loss)
                
                # Validation phase
                if val_graphs:
                    self.model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for graph in val_graphs:
                            graph = graph.to(self.device)
                            out = self.model(graph.x, graph.edge_index)
                            loss = criterion(out, graph.y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_graphs)
                    val_losses.append(val_loss)
                    
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
                else:
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            history = {'train_loss': train_losses}
            if val_losses:
                history['val_loss'] = val_losses
            
            return history
        
        def predict(self, whisker_data: pd.DataFrame) -> np.ndarray:
            """
            Predict whisker IDs for each frame.
            
            Args:
                whisker_data: Input data
                
            Returns:
                Predicted whisker IDs matching original data order
            """
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            # Prepare features
            processed_data = self._prepare_features(whisker_data)
            
            # Build graphs
            graphs, _, _ = build_whisker_graph(processed_data, self.temporal_window)
            
            if len(graphs) == 0:
                return processed_data['wid'].values
            
            # Create mapping from frame to predictions
            frames = sorted(whisker_data['fid'].unique())
            frame_predictions = {}
            
            self.model.eval()
            with torch.no_grad():
                for i, graph in enumerate(graphs):
                    if i >= len(frames):
                        break
                        
                    graph = graph.to(self.device)
                    out = self.model(graph.x, graph.edge_index)
                    predictions = torch.argmax(out, dim=1)
                    
                    # Get the target frame (center of temporal window)
                    target_frame = frames[i]
                    
                    # Get data for this specific frame from the graph
                    frame_data = whisker_data[whisker_data['fid'] == target_frame]
                    
                    # Map predictions back to original whisker IDs
                    # We need to match the graph node order with the frame data order
                    if len(predictions) >= len(frame_data):
                        # Take the first predictions corresponding to this frame
                        frame_preds = predictions[:len(frame_data)]
                        mapped_preds = [self.class_to_id[pred.item()] for pred in frame_preds]
                        frame_predictions[target_frame] = mapped_preds
                    else:
                        # Fallback to original IDs if prediction count mismatch
                        frame_predictions[target_frame] = frame_data['wid'].tolist()
            
            # Reconstruct predictions in original data order
            result_predictions = []
            for _, row in whisker_data.iterrows():
                frame_id = row['fid']
                if frame_id in frame_predictions:
                    # Get prediction for this whisker in this frame
                    frame_data = whisker_data[whisker_data['fid'] == frame_id]
                    whisker_idx = frame_data.index.get_loc(row.name)
                    if whisker_idx < len(frame_predictions[frame_id]):
                        result_predictions.append(frame_predictions[frame_id][whisker_idx])
                    else:
                        result_predictions.append(row['wid'])
                else:
                    result_predictions.append(row['wid'])
            
            return np.array(result_predictions)
        
        def save_model(self, filepath: str):
            """Save the trained model."""
            if self.model is None:
                raise ValueError("No model to save")
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'n_whiskers': self.n_whiskers,
                'temporal_window': self.temporal_window,
                'input_dim': self.input_dim,
                'id_to_class': self.id_to_class,
                'class_to_id': self.class_to_id
            }, filepath)
        
        def load_model(self, filepath: str):
            """Load a trained model."""
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.n_whiskers = checkpoint['n_whiskers']
            self.temporal_window = checkpoint['temporal_window']
            self.input_dim = checkpoint['input_dim']
            self.scaler = checkpoint['scaler']
            self.id_to_class = checkpoint.get('id_to_class', {})
            self.class_to_id = checkpoint.get('class_to_id', {})
            
            self.model = WhiskerGCN(
                input_dim=self.input_dim,
                hidden_dim=64,
                num_classes=self.n_whiskers
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
else:
    # Dummy class when torch is not available
    class GNNWhiskerTracker:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for GNNWhiskerTracker")


def reassign_whisker_ids_gnn(whisker_data: pd.DataFrame, 
                             model_path: Optional[str] = None,
                             n_epochs: int = 100,
                             temporal_window: int = 10,
                             train_split: float = 0.8,
                             verbose: bool = True) -> Tuple[pd.DataFrame, bool]:
    """
    Reassign whisker IDs using GNN approach with fallback for missing dependencies.
    
    Args:
        whisker_data: DataFrame with whisker tracking data
        model_path: Path to pretrained model (if available)
        n_epochs: Number of training epochs
        temporal_window: Temporal window size
        train_split: Training split ratio
        verbose: Whether to print progress
        
    Returns:
        Tuple of (DataFrame with reassigned whisker IDs, whether GNN was used)
    """
    if not TORCH_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        warnings.warn(
            "PyTorch and/or torch_geometric not available. "
            "Using fallback ID reassignment based on spatial consistency."
        )
        return _fallback_reassignment(whisker_data, temporal_window, verbose), False
    
    # Check if sklearn is available for preprocessing
    if not SKLEARN_AVAILABLE:
        warnings.warn("sklearn not available. Using basic preprocessing.")
    
    try:
        if verbose:
            print("Initializing GNN-based whisker ID reassignment...")
        
        # Use the full GNN implementation
        tracker = GNNWhiskerTracker(
            n_whiskers=whisker_data['wid'].nunique(),
            temporal_window=temporal_window
        )
        
        # Train model if no pretrained model is provided
        if model_path is None or not os.path.exists(model_path):
            if verbose:
                print("Training new GNN model...")
            tracker.train(whisker_data, n_epochs=n_epochs, train_split=train_split, verbose=verbose)
        else:
            if verbose:
                print(f"Loading pretrained model from {model_path}")
            tracker.load_model(model_path)
        
        # Predict new IDs
        new_ids = tracker.predict(whisker_data)
        
        # Create output DataFrame and map predictions
        result_df = whisker_data.copy()
        result_df['wid'] = new_ids
        
        if verbose:
            diff = (whisker_data['wid'] != result_df['wid']).sum()
            print(f"GNN ID reassignment completed. Changed {diff} IDs.")
        
        return result_df, True
        
    except Exception as e:
        warnings.warn(f"GNN reassignment failed: {e}. Using fallback method.")
        return _fallback_reassignment(whisker_data, temporal_window, verbose), False


def _fallback_reassignment(whisker_data: pd.DataFrame, 
                          temporal_window: int = 10,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Fallback ID reassignment based on spatial consistency when GNN is not available.
    
    This method uses a simple approach based on spatial distance to maintain
    temporal consistency of whisker IDs.
    """
    if verbose:
        print("Using fallback ID reassignment based on spatial consistency...")
    
    # In-place fallback using spatial consistency
    result_df = whisker_data.copy()
    # Process each face side separately
    for face_side in result_df['face_side'].unique():
        mask_side = result_df['face_side'] == face_side
        frames = sorted(result_df.loc[mask_side, 'fid'].unique())
        for i in range(1, len(frames)):
            prev_f = frames[i-1]
            curr_f = frames[i]
            mask_prev = mask_side & (result_df['fid'] == prev_f)
            mask_curr = mask_side & (result_df['fid'] == curr_f)
            prev_rows = result_df.loc[mask_prev]
            curr_rows = result_df.loc[mask_curr]
            if prev_rows.empty or curr_rows.empty:
                continue
            # Positions
            prev_pos = prev_rows[['follicle_x','follicle_y']].values
            curr_pos = curr_rows[['follicle_x','follicle_y']].values
            # Distance matrix
            dists = np.linalg.norm(curr_pos[:, None, :] - prev_pos[None, :, :], axis=2)
            # Greedy assignment
            used = set()
            new_ids = []
            for j in range(dists.shape[0]):
                # choose nearest unused
                choices = [(d, idx) for idx, d in enumerate(dists[j]) if idx not in used]
                if choices:
                    _, best = min(choices)
                    used.add(best)
                    new_ids.append(prev_rows.iloc[best]['wid'])
                else:
                    new_ids.append(curr_rows.iloc[j]['wid'])
            # Assign new IDs
            for idx, wid in zip(curr_rows.index, new_ids):
                result_df.at[idx, 'wid'] = wid
    if verbose:
        print("Fallback ID reassignment completed.")
    return result_df


if __name__ == "__main__":

    print("GNN Whisker Classifier")
    print("======================")
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Please install PyTorch to use GNN-based tracking.")
        exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Reclassify whisker tracking data")
    parser.add_argument("file_path", type=str, help="Path to the input Parquet file")
    # parser.add_argument("protraction_direction", type=str, help="Path to the whiskerpad JSON file, or the protraction direction itself")
    parser.add_argument("--plot", action="store_true", help="Plot the whisker traces before and after reclassification")
    args = parser.parse_args()

    whisker_data = pd.read_parquet(args.file_path)

    # Run GNN tracker
    print(f"Running GNN-based whisker ID reassignment on {args.file_path}")
    result_df, used_gnn = reassign_whisker_ids_gnn(whisker_data, n_epochs=20, verbose=True)
    
    if result_df is None:
        print("ERROR: Whisker ID reassignment failed completely.")
        exit(1)
    elif used_gnn:
        print("SUCCESS: GNN-based whisker ID reassignment completed successfully!")
    else:
        print("SUCCESS: Fallback spatial consistency reassignment completed successfully!")
        print("Note: GNN method failed, so fallback method was used instead.")
