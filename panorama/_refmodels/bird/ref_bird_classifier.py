from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import cv2
import time
import numpy as np
import sys
import os
sys.path.append(
    os.path.dirname(
        os.path.realpath(
            __file__
        )
    )
)
from nets import inception
from panorama.misctools.IO import load_obj
import inception_preprocessing
from label_mapping import LABEL_CLASS
slim = tf.contrib.slim


class RefBirdClassifier(object):
    def __init__(self,
                 cnn_model_path=None,
                 lr_model_path=None,
                 center_crop=True):
        self.inception_dim = 299
        self.arg_scope = inception.inception_v3_arg_scope()
        self.endpoint = 'Mixed_7c'
        self.moving_average_decay = 0.9999
        self.center_crop = center_crop
        self.label_class = LABEL_CLASS
        self.graph = tf.Graph()
        if cnn_model_path:
            self.init_cnn_graph(cnn_model_path)
        if lr_model_path:
            self.init_lr_model(lr_model_path)
        config_sess = tf.ConfigProto(allow_soft_placement=True)
        config_sess.gpu_options.allow_growth = True
        self.sess = tf.Session(
            config=config_sess, graph=self.graph)
        self.init_fn(self.sess)

    def init_cnn_graph(self, checkpoints_path):
        with self.graph.as_default():
            tf_global_step = tf.train.get_or_create_global_step()
            self.image_input = tf.placeholder(
                tf.float32, shape=(None, None, 3))
            image = inception_preprocessing.preprocess_image(
                self.image_input,
                self.inception_dim,
                self.inception_dim,
                is_training=False,
                center_crop=self.center_crop,
            )
            images = tf.expand_dims(image, 0)

            with slim.arg_scope(self.arg_scope):
                slim_args = [slim.batch_norm, slim.dropout]
                with slim.arg_scope(slim_args, is_training=False):
                    with tf.variable_scope('InceptionV3', reuse=None) as scope:
                        net, _ = inception.inception_v3_base(
                            images, final_endpoint=self.endpoint, scope=scope)
            self.net = tf.reduce_mean(net, [0, 1, 2])

            variable_averages = tf.train.ExponentialMovingAverage(
                self.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore()
            self.init_fn = slim.assign_from_checkpoint_fn(
                checkpoints_path, variables_to_restore)

    def init_lr_model(self, lr_model_path):
        self.LR = load_obj(lr_model_path)

    def extract_features(self, image_data):
        """
        input: RGB image 0-1 range, float
        """
        begin_time = time.time()
        fea = self.sess.run(self.net, feed_dict={self.image_input: image_data})
        curr_dur = time.time() - begin_time
        return fea, curr_dur

    def load_from_disk(self, image_path):
        image_data = cv2.imread(image_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = image_data / 255.
        image_data = np.float32(image_data)
        return image_data

    def load_from_pil(self, pil_img):
        image_data = np.array(pil_img, dtype=np.float32)
        image_data = image_data / 255.
        return image_data

    def predict(self, image_data, timing=False):
        fea, dur = self.extract_features(image_data)
        predicted_proba = self.LR.predict_proba([fea])[0]
        predicted_label = np.argmax(predicted_proba)
        predicted_score = predicted_proba[predicted_label]
        predicted_class = self.LR.classes_[predicted_label]
        if timing:
            return dur, predicted_score, predicted_class
        else:
            return predicted_score, predicted_class
