#!/usr/bin/env bash
FACE_TRAIN_ANN='../../dataset/faces/data/pascal/train/in/Annotations'
FACE_TRAIN_IMG='../../dataset/faces/data/pascal/train/in/JPEGImages'
FACE_VALID_ANN='../../dataset/faces/data/pascal/val/in/Annotations'
FACE_VALID_IMG='../../dataset/faces/data/pascal/val/in/JPEGImages'
LOG_DIR='../../logs'
MODEL_DIR='../../trained_models'
python train.py \
--config_savedir ./faces_config.json \
--train_annot_folder ${FACE_TRAIN_ANN} \
--train_image_folder ${FACE_TRAIN_IMG} \
--valid_annot_folder ${FACE_VALID_ANN} \
--valid_image_folder ${FACE_VALID_IMG} \
--log_dir ${LOG_DIR}/panorama_faces \
--model_save_path ${MODEL_DIR}/panorama_faces.h5 \
--batch_size 64 \
--learning_rate 0.5e-4 \
--loss_weights 8.0 2.0 1.0