#!/usr/bin/env bash
MODEL_DIR='../../trained_models'
MQ_PATH='../../trained_models/panorama_faces_mq.csv'
python model_qualification.py \
--config_savedir ./faces_config.json \
--model_save_path ${MODEL_DIR}/panorama_faces.h5 \
--root ${DATA_ROOT} \
--save_path ${MQ_PATH}
