#!/usr/bin/env bash
VIDEO_ROOT="../../dataset/faces/video"
VIDEO_NAME="faces.flv"
FACES_ROOT="../../dataset/faces/raw"
MTCNN_WEIGHTS='../../trained_models/align'
FACENET_WEIGHTS="../../trained_models/20170512-110547/20170512-110547.pb"
FACES_DEST="../../dataset/faces/data"
ffmpeg -i "$VIDEO_ROOT/faces.flv" -vf fps=1 "$FACES_ROOT/frame_%10d.jpg" &&
python face_data_split.py \
--frames_root "$FACES_ROOT/*.jpg" \
--mtcnn_weights ${MTCNN_WEIGHTS} \
--facenet_weights ${FACENET_WEIGHTS} \
--destination ${FACES_DEST} \
--min_cluster_size 500




