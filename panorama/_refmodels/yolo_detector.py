from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
from keras.models import load_model

import keras.backend as K
import numpy as np
import cv2
from PIL import Image
import panorama._refmodels.detect_face as detect_face
import time
from panorama._refmodels.yolo.keras_yolo import yolo_head, yolo_eval


class YoloDetector(object):
    def __init__(self, model_path, obj_threshold=0.3, nms_threshold=0.3,
                 anchors=None, class_names=None):

        if class_names:
            self.class_names = class_names
        else:
            # default vocabulary, COCO
            self.class_names = ['person', 'bicycle', 'car', 'motorcycle',
                                'airplane', 'bus', 'train', 'truck', 'boat',
                                'traffic light', 'fire hydrant', 'stop sign',
                                'parking meter', 'bench', 'bird', 'cat', 'dog',
                                'horse', 'sheep', 'cow', 'elephant', 'bear',
                                'zebra', 'giraffe', 'backpack', 'umbrella',
                                'handbag', 'tie', 'suitcase', 'frisbee',
                                'skis', 'snowboard', 'sports ball', 'kite',
                                'baseball bat', 'baseball glove', 'skateboard',
                                'surfboard', 'tennis racket', 'bottle',
                                'wine glass', 'cup', 'fork', 'knife',
                                'spoon', 'bowl', 'banana', 'apple',
                                'sandwich', 'orange', 'broccoli', 'carrot',
                                'hot dog', 'pizza', 'donut', 'cake', 'chair',
                                'couch', 'potted plant', 'bed', 'dining table',
                                'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                'keyboard', 'cell phone', 'microwave', 'oven',
                                'toaster', 'sink', 'refrigerator', 'book',
                                'clock', 'vase', 'scissors', 'teddy bear',
                                'hair drier', 'toothbrush']
        if anchors:
            self.anchors = anchors
        else:
            # Default anchors
            self.anchors = np.array([[0.57273, 0.677385],
                                     [1.87446, 2.06253],
                                     [3.33843, 5.47434],
                                     [7.88282, 3.52778],
                                     [9.77052, 9.16828]])

        self.model_path = model_path
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.init_cnn(model_path, obj_threshold=obj_threshold,
                      nms_threshold=nms_threshold)

    def init_cnn(self, model_path, obj_threshold=0.3, nms_threshold=0.3):
        self.num_classes = len(self.class_names)
        self.num_anchors = len(self.anchors)
        self.net = load_model(model_path)
        self.sess = K.get_session()
        self.model_output_channels = self.net.layers[-1].output_shape[-1]
        self.yolo_outputs = yolo_head(
            self.net.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2, ))
        self.model_image_size = self.net.layers[0].input_shape[1:3]
        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo_outputs,
            self.input_image_shape,
            score_threshold=obj_threshold,
            iou_threshold=nms_threshold
        )

    def predict(self, image, timing=False):
        """Use PIL for loading data"""
        image_data = self.preprocessed(image)
        begin = time.time()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.net.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        end = time.time()
        dur = end - begin
        ture_out_boxes = []
        for out_box in out_boxes:
            ture_out_boxes.append(
                [
                    out_box[1],
                    out_box[0],
                    out_box[3],
                    out_box[2]
                ]
            )
        out_boxes = np.array(ture_out_boxes)
        if timing:
            return dur, out_boxes, out_scores, out_classes
        else:
            return out_boxes, out_scores, out_classes

    def load_from_disk(self, image_path):
        image = Image.open(image_path)
        return image

    def preprocessed(self, image):
        resized_image = image.resize(
            tuple(reversed(self.model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')[..., :3]
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        return image_data
