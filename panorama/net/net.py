from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from panorama.net.zoo.yolotiny import yolo_tiny_cascade
from panorama.net.zoo.yolotiny import custom_loss_deco
from panorama.preprocessing import parse_annotation
from panorama.misctools.utils import sep_same_nonsame
from panorama.misctools.IO import load_obj
from panorama.preprocessing import BatchGenerator
from panorama.preprocessing import std_normalize
from panorama.net.utils import time_it
from panorama.net.utils import top_n_res
from panorama.net.utils import decode_netout
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import random
import numpy as np
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import cv2
import time
from keras import backend as K
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance_matrix
import tensorflow as tf
MQ_SCHEMA = ["obj_thr",
             "ver_gamma", "depth",
             "target", "seed", "total_same", "total_nonsame",
                       "total_detected_same", "total_detected_nonsame",
                       "acc_sm", "acc_nsm", "acc", "pre", "rcl", "ta_same",
                       "tr_nonsame", "low", "high", "IDK", "GT_return"]
VAR_SCHEMA = MQ_SCHEMA[:5]


class KNNBase(object):
    def predict(self, embs, k, raw=False, wbb=False, out_boxes=None, out_scores=None):
        start = time.time()
        dists, inds = self.kneighbors(embs, n_neighbors=k)
        dur = time.time() - start
        if wbb:
            curr_res = zip(np.asarray(self.y_vals_train)[
                inds].flatten(), dists.flatten(), out_boxes, out_scores, embs)
            curr_res_raw = zip(np.asarray(self.y_vals_train)[
                inds], dists, out_boxes, out_scores, embs)
            curr_res = top_n_res(curr_res, k, reverse=False, wbb=wbb)
        else:
            curr_res_raw = zip(np.asarray(self.y_vals_train)[
                inds], dists)
            curr_res = zip(np.asarray(self.y_vals_train)[
                inds].flatten(), dists.flatten())
            curr_res = top_n_res(curr_res, k, reverse=False)
        res_set = set([x[0] for x in curr_res])
        if raw:
            return dur, res_set, curr_res, curr_res_raw
        else:
            return dur, res_set, curr_res


class K_nn_cpu(KNNBase):
    def __init__(self, nnei, alg='brute', X=None, y=None):
        self.nnei = nnei
        self.alg = alg
        self.x_vals_train = X
        self.y_vals_train = y
        self.neigh = \
            KNeighborsClassifier(n_neighbors=nnei, algorithm=alg)

        if X and y:
            self.fit(X, y)

    def fit(self, X, y=None):
        self.x_vals_train = np.array(X)
        self.y_vals_train = y

        self.neigh.fit(self.x_vals_train, range(len(self.y_vals_train)))

    def kneighbors(self, embs, n_neighbors=5):
        start = time.time()
        result = self.neigh.kneighbors(embs, n_neighbors)
        dur = time.time() - start
        return result


class K_nn(KNNBase):
    def __init__(self, nnei, alg='brute', X=None, y=None, cpu=False):
        self.nnei = nnei
        self.sess = tf.Session()
        self.x_vals_train = np.array([])
        self.y_vals_train = np.array([])
        self.inited = False
        self.cpu = cpu
        if X and y:
            self.fit(X, y)

    def kneighbors(self, embs, n_neighbors=5):
        if(n_neighbors != self.nnei):
            top_k_dists, top_k_indices = tf.nn.top_k(
                tf.negative(self.distance), k=n_neighbors)
            top_k_dists = tf.negative(top_k_dists)
            res = top_k_dists, top_k_indices
            return self.sess. \
                run(res,
                    feed_dict={self.x_data_train: self.x_vals_train,
                               self.x_data_test: embs})
        else:
            start = time.time()
            result = self.sess.run(self.res, feed_dict={
                                   self.x_data_train: self.x_vals_train, self.x_data_test: embs})
            dur = time.time() - start
#             print (dur)
            return result

    def update(self, X, y):
        if self.x_vals_train.size and self.y_vals_train.size:
            X_new = np.append(self.x_vals_train, X, axis=0)
            y_new = np.append(self.y_vals_train, y, axis=0)
        else:
            X_new = np.asarray(X)
            y_new = np.asarray(y)
        self.fit(X_new, y_new)

    def fit(self, X, y):
        self.x_vals_train = np.asarray(X)
        self.y_vals_train = np.asarray(y)
        if not self.inited:
            feature_number = len(self.x_vals_train[0])
            self.x_data_train = tf.placeholder(
                shape=[None, feature_number], dtype=tf.float32)
            self.x_data_test = tf.placeholder(
                shape=[None, feature_number], dtype=tf.float32)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(
                self.x_data_train, tf.expand_dims(self.x_data_test, 1))), axis=2))
            self.top_k_dists, self.top_k_indices = tf.nn.top_k(
                tf.negative(self.distance), k=self.nnei)
            self.top_k_dists = tf.negative(self.top_k_dists)
            self.res = self.top_k_dists, self.top_k_indices
            self.inited = True


# def k_nn(nnei,alg,X_res,y_res):
#     neigh=KNeighborsClassifier(n_neighbors=nnei,algorithm=alg)
#     neigh.fit(X_res, y_res)
#     return neigh


class PanoramaNet(object):
    def __init__(self,
                 config,
                 architecture='yolotiny',
                 normalize=std_normalize,
                 input_tensor=None,
                 aux_dataset=None):
        if architecture != 'yolotiny':
            raise NotImplementedError(
                "No architecture other than 'yolotiny' supported so far")
        self.TRUE_BOX_BUFFER = config['TRUE_BOX_BUFFER']
        self.IMAGE_H = config['IMAGE_H']
        self.IMAGE_W = config['IMAGE_W']
        self.GRID_H = config['GRID_H']
        self.GRID_W = config['GRID_W']
        self.BOX = config['BOX']
        self.CLASS = config['CLASS']
        self.ANCHORS = config['ANCHORS']
        self.config = config
        self.var_schema = VAR_SCHEMA
        self.net, \
            self.input_image, \
            self._true_boxes = yolo_tiny_cascade(
                self.GRID_H,
                self.GRID_W,
                self.BOX,
                self.CLASS,
                (self.IMAGE_H, self.IMAGE_W, 3),
                (1, 1, 1, self.TRUE_BOX_BUFFER, 4),
                input_tensor=input_tensor
            )
        self.depth_ls = range(3)
        self.layer_name = {
            0: ['l2_norm_layer_0', 'lambda_1'],
            1: ['l2_norm_layer_1', 'lambda_2'],
            2: ['l2_norm_layer_2', 'lambda_3']
        }
        self.layer_name_in = {
            0: ['input_1'],
            1: ['max_pooling2d_6'],
            2: ['conv_8']
        }
        self.layer_name_out = {
            0: ['leaky_re_lu_6', 'l2_norm_layer_0', 'lambda_1'],
            1: ['leaky_re_lu_7', 'l2_norm_layer_1', 'lambda_2'],
            2: ['leaky_re_lu_8', 'l2_norm_layer_2', 'lambda_3']
        }
        self.ops_dict = {}
        self.lhls = {}
        self.submodels = {}
        self.dummy_array = np.zeros(
            (1, 1, 1, 1, config['TRUE_BOX_BUFFER'], 4))
        self.cascade_init()
        self.graph = None
        self.normalize = normalize
        if aux_dataset:
            self.aux_dataset = aux_dataset

    def cascade_init(self):
        all_emb_grids = []
        all_netouts = []
        for depth in self.depth_ls:
            emb_name, output_name = self.layer_name[depth]
            emb_grid = self.net.get_layer(emb_name).output
            netout = self.net.get_layer(output_name).output
            all_emb_grids.append(emb_grid)
            all_netouts.append(netout)
            new_model = Model(inputs=self.net.input,
                              outputs=[emb_grid, netout])
            self.submodels[depth] = new_model
        all_outputs = zip(all_emb_grids, all_netouts)
        all_outputs = [item for sublist in all_outputs for item in sublist]
        self.submodels['all'] = Model(inputs=self.net.input,
                                      outputs=all_outputs)
        # Prepare inter-operators
        model = self.net
        layer_name_in = self.layer_name_in
        layer_name_out = self.layer_name_out
        true_boxes = model.input[1]
        outputs_all = []
        inputs_first = [model.input[0], true_boxes]
        for depth in self.depth_ls:
            layer_name_in = self.layer_name_in[depth]
            layer_name_out = self.layer_name_out[depth]
            inputs = inputs_first if depth == 0 else \
                [model.get_layer(layer_name_in[0]).input, true_boxes]
            outputs = [model.get_layer(
                lname).output for lname in layer_name_out]
            outputs_all += outputs
            sub_op = time_it(K.function(inputs, outputs))
            self.ops_dict[depth] = sub_op
        self.get_out_all = time_it(K.function(inputs_first, outputs_all))

    def set_cas_param(self,
                      model_qualification_path,
                      obj_thr=None, ver_gamma=None, target=None, inspection=False, auto=True):
        mq_ori = pd.read_csv(model_qualification_path)
        mq = mq_ori.set_index(self.var_schema).sort_index()
        mq_gb = mq.groupby(self.var_schema[:-1]).mean().reset_index()
        mq_gb['combined'] = mq_gb['IDK'] + mq_gb['GT_return']
        if inspection:
            return mq_gb.groupby(['obj_thr', 'ver_gamma', 'target']) \
                .mean().sort_values('combined')
        if auto:
            df_sorted = mq_gb.groupby(['obj_thr', 'ver_gamma', 'target']) \
                .mean().sort_values('combined').reset_index()
            head = df_sorted.iloc[0]
            obj_thr, ver_gamma, target = \
                head['obj_thr'], head['ver_gamma'], head['target']
            # print(obj_thr, ver_gamma, target)
            self.obj_thr = obj_thr
            self.ver_gamma = ver_gamma
            self.target = target
        mq_gb = mq_gb.sort_values('combined')
        df = mq_gb[
            (mq_gb['obj_thr'] == obj_thr) & (mq_gb['ver_gamma'] == ver_gamma)
        ]
        for depth in self.depth_ls:
            row = df[(df['depth'] == depth) & (df['target'] == target)]
            low = row['low'].iloc[0]
            high = row['high'].iloc[0]
            self.lhls[depth] = [low, high]

    def set_slacks(self, slacks):
        '''Set the slack variables'''
        for depth, slack in zip(self.depth_ls, slacks):
            self.lhls[depth][0] -= slack
            self.lhls[depth][1] += slack

    def train(self,
              log_dir,
              model_save_path,
              learning_rate=0.5e-4,
              batch_size=8,
              warm_up_batches=0,
              epochs=150,
              patience=10,
              loss_weights=[8.0, 2.0, 1.0],
              save_all=False,
              checkpoint='',
              clipnorm=None):
        model = self.net
        true_boxes = self._true_boxes
        custom_loss = custom_loss_deco(self.config['GRID_H'],
                                       self.config['GRID_W'],
                                       batch_size,
                                       self.config['ANCHORS'],
                                       self.config['BOX'],
                                       self.config['COORD_SCALE'],
                                       self.config['NO_OBJECT_SCALE'],
                                       self.config['OBJECT_SCALE'],
                                       self.config['CLASS_WEIGHTS'],
                                       self.config['CLASS_SCALE'],
                                       self.config['CLASS_SCALE'],
                                       true_boxes)
        if os.path.exists(checkpoint):
            print ('Checkpoints found, loading')
            model.load_weights(checkpoint,
                               by_name=True, skip_mismatch=True)
        generator_config = {
            'IMAGE_H': self.config['IMAGE_H'],
            'IMAGE_W': self.config['IMAGE_W'],
            'GRID_H': self.config['GRID_H'],
            'GRID_W': self.config['GRID_W'],
            'BOX': self.config['BOX'],
            'LABELS': self.config['LABELS'],
            'CLASS': self.config['CLASS'],
            'ANCHORS': self.config['ANCHORS'],
            'BATCH_SIZE': batch_size,
            'TRUE_BOX_BUFFER': 50,
            'CASCADE_LEVEL': 3,
        }
        train_imgs, seen_train_labels = parse_annotation(
            self.config['train_annot_folder'],
            self.config['train_image_folder'],
            labels=self.config['LABELS']
        )
        valid_imgs, seen_valid_labels = parse_annotation(
            self.config['valid_annot_folder'],
            self.config['valid_image_folder'],
            labels=self.config['LABELS']
        )
        random.shuffle(valid_imgs)
        random.shuffle(train_imgs)

        # train_imgs, valid_imgs = list(percentage_split(
        #     train_imgs, [1 - validation_split, validation_split]))
        train_batch = BatchGenerator(
            train_imgs, generator_config, norm=self.normalize)
        valid_batch = BatchGenerator(
            valid_imgs, generator_config, norm=self.normalize)
        if clipnorm:
            optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                             epsilon=1e-08, decay=0.0, clipnorm=clipnorm)
        else:
            optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                             epsilon=1e-08, decay=0.0)
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.001,
                                   patience=patience,
                                   mode='min',
                                   verbose=1)
        save_best_only = (not save_all)
        checkpoint = ModelCheckpoint(model_save_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=save_best_only,
                                     mode='min',
                                     period=1)
        tensorboard = TensorBoard(log_dir=log_dir,
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)
        model.compile(loss=custom_loss, optimizer=optimizer,
                      loss_weights=loss_weights)

        model.fit_generator(generator=train_batch,
                            steps_per_epoch=len(train_batch),
                            epochs=epochs,
                            verbose=1,
                            validation_data=valid_batch,
                            validation_steps=len(valid_batch),
                            callbacks=[early_stop, checkpoint, tensorboard],
                            max_queue_size=5)

    def load_weights(self, model_save_path=None, by_name=False):
        if not model_save_path:
            model_save_path = self.config['model_save_path']
        print("Loading weights from {}".format(model_save_path))
        self.net.load_weights(model_save_path, by_name=by_name)

    def summary(self):
        return self.net.summary()

    def poll_embs(self, emb_grid, boxes):
        return [emb_grid[0][w][h][b] for box, (w, h, b) in boxes]

    def load_from_disk(self, image_path):
        image_data = cv2.imread(image_path)
        return image_data

    def get_raw(self, image_data, cascade_depth):
        image_h, image_w, _ = image_data.shape
        image_data = self.preprocessed(image_data)
        begin = time.time()
        out = self.submodels[cascade_depth].predict(
            [image_data, self.dummy_array])
        dur = time.time() - begin
        if cascade_depth == 'all':
            emb_grid = out[0::2]
            netout = out[1::2]
        else:
            emb_grid, netout = out
        return dur, emb_grid, netout, image_h, image_w

    def decode_raw(self,
                   raw,
                   obj_threshold,
                   nms_threshold,
                   obj_class
                   ):
        emb_grid, netout, image_h, image_w = raw
        boxes = decode_netout(netout[0],
                              obj_threshold=obj_threshold,
                              nms_threshold=nms_threshold,
                              anchors=self.ANCHORS,
                              nb_class=self.CLASS,
                              obj_class=obj_class)
        out_boxes = []
        out_scores = []
        out_classes = []
        for box in boxes:
            outbox, out_score, out_class = self.decode_box(
                image_h, image_w, box, obj_class)
            out_boxes.append(outbox)
            out_scores.append(out_score)
            out_classes.append(out_class)
        embs = self.poll_embs(emb_grid, boxes)
        out_boxes = np.array(out_boxes)
        out_scores = np.array(out_scores)
        out_classes = np.array(out_classes)
        idx = np.argsort(out_scores)[::-1]
        out_scores = out_scores[idx]
        out_classes = out_classes[idx]
        out_boxes = out_boxes[idx]
        embs = list(np.array(embs)[idx])
        return out_boxes, \
            out_scores, \
            out_classes, \
            embs

    def predict(self,
                image_data,
                obj_threshold,
                nms_threshold,
                cascade_depth=-1,
                obj_class=False,
                return_raw=False,
                k=None):
        raw = self.get_raw(image_data, cascade_depth)
        dur = raw[0]
        decoded = self.decode_raw(
            raw[1:], obj_threshold, nms_threshold, obj_class)
        if return_raw:
            return dur, decoded
        else:
            (out_boxes,
             out_scores,
             out_classes,
             embs) = decoded
            out_labels = np.array(self.config['LABELS'])[out_classes]
            curr_res = top_n_res(zip(out_labels, out_scores), k, reverse=True)
            res_set = set([x[0] for x in curr_res])
            return dur, curr_res, res_set

    def ensemble_predict_decode(self,
                                netouts, image_h, image_w,
                                obj_threshold, nms_threshold, obj_class=False):
        out_boxes_ensemble = []
        out_scores_ensemble = []
        out_classes_ensemble = []
        for netout in netouts:
            boxes = decode_netout(netout[0],
                                  obj_threshold=obj_threshold,
                                  nms_threshold=nms_threshold,
                                  anchors=self.ANCHORS,
                                  nb_class=self.CLASS,
                                  obj_class=obj_class)
            out_boxes = []
            out_scores = []
            out_classes = []
            for box in boxes:
                outbox, out_score, out_class = self.decode_box(
                    image_h, image_w, box, obj_class)
                out_boxes.append(outbox)
                out_scores.append(out_score)
                out_classes.append(out_class)
            out_boxes_ensemble.append(out_boxes)
            out_scores_ensemble.append(out_scores)
            out_classes_ensemble.append(out_classes)
        return out_boxes_ensemble, out_scores_ensemble, out_classes_ensemble

    def decode_box(self, image_h, image_w, box, obj_class):
        box = box[0]
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)
        result = [np.array([ymin, xmin, ymax, xmax]),
                  box.get_score(), box.get_label()]
        if obj_class:
            result[1] = box.c
        return result

    def preprocessed(self, image):
        input_image = cv2.resize(image, (self.IMAGE_H, self.IMAGE_W))
#         input_image = input_image / 255.
        input_image = self.normalize(input_image)
        input_image = input_image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        return input_image

    def graph(self):
        if not self.graph:
            self.graph = model_to_dot(self.net).create(
                prog='dot', format='svg')
        return self.graph

    def netout_to_emb(self, netout, emb_grid, OBJ_THRESHOLD, NMS_THRESHOLD, obj_class=True):
        boxes = decode_netout(netout[0],
                              obj_threshold=OBJ_THRESHOLD,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=self.ANCHORS,
                              nb_class=self.CLASS,
                              obj_class=obj_class)
        embs = [emb_grid[0][w][h][b] for box, (w, h, b) in boxes]
        return embs

    def verify(self,
               file_pair,
               obj_thr=None,
               nms_thr=None,
               thr=None, no_GT=False, k=5, binary=True, verbose=False):
        if not obj_thr:
            obj_thr = self.obj_thr
        if not nms_thr:
            nms_thr = 0.5
        if not thr:
            thr = self.ver_gamma
        total_dur = [0] * len(self.depth_ls)
        GT_invoked = False
        min_dis = None if not no_GT else 2
        solved_in = None
        inputs_pair = {}
        for depth in self.depth_ls:
            if verbose:
                print("depth:{}".format(depth))
            get_out_func = self.ops_dict[depth]
            embs_pair = []
            outputs_pair = {}
            for filename in file_pair:
                if depth == 0:
                    image_data = self.load_from_disk(filename)
                    image_h, image_w, _ = image_data.shape
                    image_data = self.preprocessed(image_data)
                    inputs = [image_data, self.dummy_array]
                else:
                    inputs = inputs_pair[filename]
                dur, (outputs, emb_grid, netout) = get_out_func(inputs)
                total_dur[depth] += dur
                raw = (emb_grid, netout, image_h, image_w)
                _, _, _, embs = self.decode_raw(
                    raw,
                    obj_thr,
                    nms_thr,
                    no_GT
                )
                embs = embs[:k]
                embs_pair.append(embs)
                outputs_pair[filename] = outputs
                if not embs:
                    if verbose:
                        print("Not detected")
                    GT_invoked = True
                    if not no_GT:
                        break
            if GT_invoked:
                if not no_GT:
                    break
                else:
                    if depth == self.depth_ls[-1]:
                        if verbose:
                            print ('Early termination for depth:{}, min_dis:{}'.format(
                                depth, min_dis))
                        solved_in = depth
                        break
                    inputs_pair = {}
                    for filename, outputs in outputs_pair.items():
                        inputs_pair[filename] = [
                            outputs_pair[filename], self.dummy_array]
                    if verbose:
                        print (inputs_pair.keys())
                    continue
            else:
                min_dis = np.min(distance_matrix(
                    embs_pair[0], embs_pair[1]))
                if verbose:
                    print("min_dis:{}".format(min_dis))
                if self.lhls[depth][0] <= min_dis \
                        and min_dis < self.lhls[depth][1]:
                    if verbose:
                        print ('IDK for depth:{}'.format(depth))
                    if depth == self.depth_ls[-1]:
                        if not no_GT:
                            if verbose:
                                print("Invoke GTCNN at last depth")
                            GT_invoked = True
                        else:
                            if verbose:
                                print ('Termination for depth:{}, min_dis:{}'.format(
                                    depth, min_dis))
                            solved_in = depth
                        break
                    inputs_pair = {}
                    for filename, outputs in outputs_pair.items():
                        inputs_pair[filename] = [
                            outputs_pair[filename], self.dummy_array]

                else:
                    if verbose:
                        print ('Early termination for depth:{}'.format(depth))
                    solved_in = depth
                    break
        if GT_invoked and not no_GT:
            solved_in = 'GT'
        if binary:
            return min_dis < thr, GT_invoked, solved_in
        else:
            return total_dur, min_dis, GT_invoked, solved_in

    def load_album(self, album_path, voc, k, n=5, dup=None, K_nn=K_nn):
        label_filename_embs = load_obj(album_path)
        album = {}
        album_embs = {}
        self.album_filename_list = []
        for label, values in label_filename_embs.items():
            values.sort(key=lambda x: x[1], reverse=True)
            best_values = values[:n]
            embs = []
            for x in best_values:
                embs.append([y[2] for y in x[2]])
            album[label] = values[:n]
            album_embs[label] = embs
            self.album_filename_list.append(x[0])
        X_ad = [[] for i in range(3)]
        y_ad = [[] for i in range(3)]
        if not voc:
            print("No voc given")
            voc = set(album_embs.keys())
        for label, files in album_embs.items():
            if label in voc:
                for embs in files:
                    for i, emb in enumerate(embs):
                        if emb is not None:
                            X_ad[i].append(emb)
                            y_ad[i].append(label)
        if dup:
            X_ad = [[item for item in x for _ in range(dup)] for x in X_ad]
            y_ad = [[item for item in x for _ in range(dup)] for x in y_ad]

        neigh = [K_nn(k, 'brute', X_ad[i], y_ad[i])
                 for i in range(len(self.depth_ls))]
        new_ovoc = set([item for sublist in y_ad for item in sublist])
        return neigh, new_ovoc

    def recognize(self,
                  img_file,
                  neigh,
                  obj_thr=None,
                  nms_thr=None,
                  no_GT=True,
                  k=5,
                  l_only=None,
                  icaches=None,
                  dcache=None,
                  cache_skip=1,
                  cached_data=None,
                  data_passed=False,
                  wbb=False,
                  raw_output=False,
                  verbose=False
                  ):
        if not obj_thr:
            obj_thr = self.obj_thr
        if not nms_thr:
            nms_thr = 0.5
        inputs_pre = None
        total_dur = [0] * (len(self.depth_ls) + 1)
        GT_invoked = False
        solved_in = None
        curr_res = None
        res_set = None
        depth_embs = {}
        depth_curr_res = {}
        depth_res_set = {}
        total_embs = 0
        cache_hit = 0

        for depth in self.depth_ls:
            if verbose:
                print("depth:{}".format(depth))
            get_out_func = self.ops_dict[depth]
            outputs_pre = None
            if depth == 0:
                if cached_data:
                    image_data, image_h, image_w = cached_data[img_file]
                    if verbose:
                        print(img_file)
                elif data_passed:
                    image_data = img_file
                    image_h, image_w, _ = image_data.shape
                    image_data = self.preprocessed(image_data)
                else:
                    image_data = self.load_from_disk(img_file)
                    if verbose:
                        print(img_file)
                    image_h, image_w, _ = image_data.shape
                    image_data = self.preprocessed(image_data)
                inputs = [image_data, self.dummy_array]
            else:
                inputs = inputs_pre
            dur, (outputs, emb_grid, netout) = get_out_func(inputs)
            total_dur[depth] += dur
            raw = (emb_grid, netout, image_h, image_w)
            out_boxes, \
                out_scores, \
                out_classes, \
                embs = self.decode_raw(
                    raw,
                    obj_thr,
                    nms_thr,
                    no_GT
                )
            depth_embs[depth] = embs
            outputs_pre = outputs
            if not embs or (l_only and depth != l_only):
                if verbose:
                    print("Not detected")
                GT_invoked = True
                if not no_GT:
                    break
                else:
                    if depth == self.depth_ls[-1]:
                        if verbose:
                            print (
                            'Early termination for depth:{}, res_set:{}'.format(
                                depth, res_set
                            )
                        )
                        solved_in = depth
                        break
                    inputs_pre = [outputs_pre, self.dummy_array]
                    continue
            else:
                if icaches:
                    dur, cache_miss_idxes, curr_res_cache = icaches[depth].search(
                        embs)
                    total_embs = len(embs)
                    cache_hit = total_embs - len(cache_miss_idxes)
                    if verbose:
                        print (dur, cache_hit, total_embs)
                    total_dur[-1] += dur
                    embs = np.asarray(embs)[cache_miss_idxes]
                    embs = list(embs)
                    if embs and cache_hit <= cache_skip * total_embs:
                        dur, res_set, curr_res_missed, curr_res_raw_missed = neigh[depth].predict(
                            embs, k, raw=True)

                    else:
                        dur, _, curr_res_missed, curr_res_raw = 0, None, [], []
                    total_dur[-1] += dur
                    curr_res = curr_res_cache + curr_res_missed
                    curr_res = top_n_res(curr_res, k, reverse=False)
                    res_set = set([x[0] for x in curr_res])
                else:
                    dur, res_set, curr_res, curr_res_raw = neigh[depth].predict(
                        embs,
                        k,
                        raw=True,
                        wbb=wbb,
                        out_boxes=out_boxes,
                        out_scores=out_scores)
                    total_dur[-1] += dur
                depth_curr_res[depth] = curr_res
                depth_res_set[depth] = res_set
                cas_hit = False
                for x in curr_res:
                    if wbb:
                        distance = x[1][0]
                    else:
                        distance = x[1]
                    if not(self.lhls[depth][0] <= distance and
                           distance < self.lhls[depth][1]):
                        cas_hit = True
                        break
                if not cas_hit:
                    if verbose:
                        print ('IDK for depth:{}'.format(depth))
                    if depth == self.depth_ls[-1]:
                        if not no_GT:
                            if verbose:
                                print("Invoke GTCNN at last depth")
                            GT_invoked = True
                        else:
                            if verbose:
                                print ('Termination for depth:{}, res_set:{}'.format(
                                depth, res_set))
                            solved_in = depth
                            if icaches and curr_res_missed:
                                icaches[depth].update(
                                    embs, curr_res_raw_missed)
                        break
                    inputs_pre = [outputs_pre, self.dummy_array]
                else:
                    if verbose:
                        print ('Early termination for depth:{}'.format(depth))
                    solved_in = depth
                    if verbose:
                        print("Curr res:{}".format(curr_res))
                    if icaches and curr_res_missed:
                        icaches[depth].update(embs, curr_res_raw_missed)
                    break
        if verbose:
            print(total_dur)
        if raw_output:
            if icaches:
                return total_dur, res_set, GT_invoked, solved_in, total_embs, cache_hit
            if wbb:
                return total_dur, curr_res, GT_invoked, solved_in
            else:
                return total_dur, res_set, GT_invoked, solved_in
        else:
            return curr_res

class LRUCache(object):
    def __init__(self, size, K_nn=K_nn):
        self.size = size
        self.queue = []
        self.knn = K_nn(1, 'brute')

    def search(self, embs):
        begin = time.time()
        curr_res = []
        cache_miss_idxes = []
        if len(self.queue) == 0:
            cache_miss_idxes = range(len(embs))
        else:
            dists, inds = self.knn.kneighbors(embs, n_neighbors=1)
            labels = np.asarray(self.y)[inds.flatten()]
            curr_res = []

            for i, (label, dist) in enumerate(zip(labels, dists.flatten())):
                min_dis = float(label[1][0])
                if dist < min_dis:
                    curr_res += zip(label[0], label[1].astype(np.float32))
                else:
                    cache_miss_idxes.append(i)

        dur = time.time() - begin
        return dur, cache_miss_idxes, curr_res

    def flattened(self):
        X = []
        y = []

        for embs, reses in self.queue:
            for emb, res in zip(embs, reses):
                X.append(emb)
                y.append(res)
        return X, y

    def update(self, embs, curr_res):
        self.queue.insert(0, [embs, curr_res])
        self.queue = self.queue[:self.size]
        self.X, self.y = self.flattened()
        self.knn.fit(self.X, self.y)


class VerificationBase(object):
    def __init__(self, panoramaNet, img_folders, ann_folders, in_labels=[]):
        self.panoramaNet = panoramaNet
        self.img_folders = img_folders
        self.ann_folders = ann_folders
        self.in_labels = in_labels
        self.labels = []
        self.file_list = []
        self.sample = 5000
        self.load_data()

    def load_data(self):
        print("Loading data ..., in labels:{}".format(self.in_labels))

        for img_folder, ann_folder in zip(self.img_folders, self.ann_folders):
            imgs, imgs_labels = parse_annotation(
                ann_folder, img_folder, self.in_labels, onlyInLabels=False
            ) if len(self.in_labels) == 0 \
                else \
                parse_annotation(
                ann_folder, img_folder, self.in_labels, onlyInLabels=True
            )
            self.file_list += imgs
            self.labels += imgs_labels

    def new_sample(self, size=8000):
        print("New sampling ... over:{}".format(len(self.file_list)))
        size = min([size, len(self.file_list)])
        file_list_sampled = random.sample(self.file_list, size)
        self.same, self.nonsame = sep_same_nonsame(
            file_list_sampled, nsame=self.sample, nnonsame=self.sample)
