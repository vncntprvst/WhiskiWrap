Train a GNN model:

python -m wwutils.classifiers.train_gnn test_videos/excerpt_video_updated_edited.parquet

Run it on the training data:
python -m wwutils.classifiers.gnn_classifier test_videos/excerpt_video_updated_edited.parquet --model wwutils/classifiers/gnn_model.pt

Run it on the not-well classified data:
python -m wwutils.classifiers.gnn_classifier test_videos/excerpt_video.parquet --model wwutils/classifiers/gnn_model.pt