# GNN-based Whisker Tracking

This module provides a Graph Neural Network (GNN) approach for improving whisker ID consistency across video frames. Instead of treating each frame independently, the GNN considers temporal relationships between whisker detections to assign more consistent IDs.

## Key Features

- **Temporal Consistency**: Uses graph structure to model whisker correspondences across frames
- **Learning-based**: Trains on existing tracking data to learn optimal assignment patterns  
- **Robust Handling**: Better handles whisker crossings and temporary occlusions
- **Integration**: Seamlessly integrates with existing WhiskiWrap pipeline

## Installation

Install additional dependencies for GNN support:

```bash
pip install -r gnn_requirements.txt
```

Or install manually:
```bash
pip install torch torch-geometric scikit-learn matplotlib
```

## Usage

### As part of the main pipeline

Replace the traditional reclassification step with GNN-based reassignment:

```bash
# Use GNN instead of reclassify step
python whisker_tracking_pipeline.py video.mp4 --use-gnn

# Train for more epochs for better accuracy
python whisker_tracking_pipeline.py video.mp4 --use-gnn --gnn-epochs 200

# Use pretrained model
python whisker_tracking_pipeline.py video.mp4 --use-gnn --gnn-model my_gnn_model.pt
```

### Standalone usage

Process already-tracked whisker data:

```bash
# Basic usage
python wwutils/classifiers/gnn_whisker_tracker.py test_videos/excerpt_video_updated_edited.parquet

# Save trained model for reuse
python wwutils/classifiers/gnn_whisker_tracker.py test_videos/excerpt_video_updated_edited.parquet --save-model my_gnn.pt

# Use custom parameters
python wwutils/classifiers/gnn_whisker_tracker.py test_videos/excerpt_video_updated_edited.parquet \
    --epochs 200 --lr 0.0005 --hidden-dim 128 --max-spatial-distance 75
```

## How it Works

1. **Graph Construction**: Builds a graph connecting whisker detections across frames based on spatial proximity and temporal adjacency

2. **Feature Extraction**: Uses whisker properties (position, angle, curvature, length) as node features

3. **GNN Training**: Trains a Graph Convolutional Network to predict correspondence confidence between whisker pairs

4. **ID Assignment**: Uses Hungarian algorithm with GNN-predicted costs to optimally assign whisker IDs

## Advantages over Traditional Methods

- **Context-aware**: Considers multiple frames simultaneously rather than frame-by-frame
- **Learning-based**: Adapts to specific whisker tracking patterns in your data
- **Robust**: Better handles challenging scenarios like whisker crossings
- **Quantitative**: Provides confidence scores for assignments

## Output

The GNN tracker generates:
- Reassigned whisker data with improved temporal consistency
- Quality metrics comparing original vs. reassigned tracking
- Visualization plots showing trajectory improvements
- Optional trained model files for reuse

## Parameters

Key parameters for tuning:

- `--epochs`: Training epochs (more = better learning, slower)
- `--lr`: Learning rate (0.001 typical, lower for fine-tuning)
- `--hidden-dim`: Model capacity (64-128 typical)
- `--max-spatial-distance`: Maximum distance for whisker connections
- `--max-frame-gap`: Maximum frame separation for temporal connections

## When to Use

GNN tracking is most beneficial for:
- Long video sequences where temporal consistency matters
- Data with frequent whisker crossings or interactions
- High-quality tracking data (GNN learns from existing assignments)
- When you need explainable assignment confidence scores

Traditional reclassification may be sufficient for:
- Short video segments
- Simple whisker movements without crossings
- Lower-quality tracking data with many errors
- Quick processing without training time
