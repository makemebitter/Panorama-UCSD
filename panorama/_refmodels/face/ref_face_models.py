from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cv2
import time
import numpy as np
import pathmagic  # noqa
import panorama._refmodels.face.detect_face as detect_face
import panorama._refmodels.face.facenet as facenet
from panorama._refmodels.face.facenet import prewhiten


slim = tf.contrib.slim


class RefFaceDetector(object):
    def __init__(self,
                 mtcnn_weights
                 ):

        self.init_cnn_graph(mtcnn_weights)
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.detection_window_size_ratio = 1 / 8  # 20/160
        self.class_names = ['face']

    def init_cnn_graph(self, checkpoints_path):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.pnet, \
                self.rnet, \
                self.onet = detect_face.create_mtcnn(
                    self.sess, checkpoints_path)

    def load_from_disk(self, image_path):
        image_data = cv2.imread(image_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return image_data

    def predict(self, image_data):
        bounding_boxes, _, duration = detect_face.detect_face(
            image_data,
            self.minsize,
            self.pnet,
            self.rnet,
            self.onet,
            self.threshold,
            self.factor)
        out_boxes = np.array([bounding_box[:4]
                              for bounding_box in bounding_boxes])
        out_scores = np.array([bounding_box[4]
                               for bounding_box in bounding_boxes])
        out_classes = np.array([0] * bounding_boxes.shape[0])

        return out_boxes, out_scores, out_classes, duration


class RefFaceExtractor(object):
    def __init__(self,
                 facenet_weights
                 ):

        self.init_cnn_graph(facenet_weights)
        self.image_size = (160, 160)

    def init_cnn_graph(self, checkpoints_path):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            print('Loading feature extraction model')
            facenet.load_model(checkpoints_path)
            self.images_placeholder = \
                tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = \
                tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]

    def load_from_disk(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = self.load_from_cv2(image)
        return image_data

    def load_from_cv2(self, image_cv2):
        image_cv2 = cv2.resize(image_cv2, self.image_size)
        image_cv2 = prewhiten(image_cv2)
        image_data = image_cv2.reshape(-1, self.image_size[0],
                                       self.image_size[1], 3)
        return image_data

    def load_from_pil(self, image_pil):
        image_data = np.array(image_pil, dtype=np.float32)
        image_data = self.load_from_cv2(image_data)
        return image_data

    def extract_features(self, image_data):

        feed_dict = {self.images_placeholder: image_data,
                     self.phase_train_placeholder: False}
        facenet_time = time.time()
        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)[0]
        duration = time.time() - facenet_time
        emb = emb.astype(np.float32)
        return emb, duration
