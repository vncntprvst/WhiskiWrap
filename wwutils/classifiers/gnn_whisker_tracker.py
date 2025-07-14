"""Deprecated module maintained for backwards compatibility.

The functionality has been split into ``gnn_classifier.py`` and ``train_gnn.py``.
Importing from this module will forward to :mod:`gnn_classifier`.
"""

from .gnn_classifier import (
    GNNWhiskerTracker,
    reassign_whisker_ids_gnn,
    assign_whisker_ids,
    extract_whisker_features,
    build_whisker_graph,
)
