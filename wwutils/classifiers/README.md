# Classifiers

This folder contains optional modules for assigning whisker IDs after the
tracking step. Three approaches are provided:

- `reclassify.py` – heuristic reassignment based on the whisker pad location.
- `unet_classifier.py` – apply a pre-trained U-Net segmentation model to each
  frame. Requires PyTorch and a trained model file.
- `gnn_classifier.py` – Graph Neural Network approach that can train on the
  tracking data itself if no model is supplied.

Utility scripts:

- `train_gnn.py` – train a GNN model from a tracking parquet file.
- `train_unet.py` – larger utility for training a U-Net model from labelled
  images or from video/parquet pairs.
- `benchmark_classifiers.py` – compare the classifiers on misclassified data.

Each classifier module exposes a small CLI. For example, to apply a GNN model:

```bash
python -m wwutils.classifiers.gnn_classifier tracking.parquet --model my_gnn.pt
```

See the docstrings of each module for additional details and options.
