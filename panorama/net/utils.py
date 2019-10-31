from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import scipy


class BoundBox(object):
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.c = c
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap(
        [box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap(
        [box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def time_it(func):
    def wrapper(*args, **kwargs):
        time_0 = time.time()
        res = func(*args, **kwargs)
        duration_0 = time.time() - time_0
        return duration_0, res

    return wrapper


def decode_netout(netout, anchors,
                  nb_class, obj_threshold=0.3,
                  nms_threshold=0.3, obj_class=False):
    """Decode the output and return the embeddings extracted
    obj_class:use object confidence or class confidence
    #TODO put into the DNN graph
    """
    grid_h, grid_w, nb_box = netout.shape[:3]
    boxes = []
    det_bbx = netout[..., :4]
    det_conf = netout[..., [4]]
    cls_conf = netout[..., 5:]
    det_conf = scipy.special.expit(det_conf)
    cls_conf = det_conf * _softmax(cls_conf)
    if obj_class:
        det_conf *= (det_conf > obj_threshold)
        idx = np.sum(det_conf, axis=-1) > 0
    else:
        cls_conf *= (cls_conf > obj_threshold)
        idx = np.sum(cls_conf, axis=-1) > 0
    cell_x = np.reshape(
        np.tile(np.arange(grid_w), [grid_h]), (grid_h, grid_w, 1, 1))
    cell_y = np.transpose(cell_x, (1, 0, 2, 3))

    cell_xy = np.concatenate([cell_x, cell_y], -1)

    cell_grid = np.tile(cell_xy, [1, 1, 5, 1])

    pred_box_xy = scipy.special.expit(det_bbx[..., :2]) + cell_grid

    pred_box_xy[..., [0]] /= grid_w
    pred_box_xy[..., [1]] /= grid_h

    anchors_shaped = np.reshape(anchors, [1, 1, nb_box, 2])

    pred_box_wh = np.exp(det_bbx[..., 2:]) * anchors_shaped
    pred_box_wh[..., [0]] /= grid_w
    pred_box_wh[..., [1]] /= grid_h

    XYMIN = pred_box_xy[idx, :] - pred_box_wh[idx, :] / 2

    XYMAX = pred_box_xy[idx, :] + pred_box_wh[idx, :] / 2

    CONF = det_conf[idx, :]

    CLASSES = cls_conf[idx, :]
    IDX = np.transpose(np.vstack(np.where(idx)))
    for i in range(CONF.shape[0]):
        box = BoundBox(XYMIN[i][0], XYMIN[i][1], XYMAX[i]
                       [0], XYMAX[i][1], CONF[i][0], CLASSES[i])
        boxes.append((box, (IDX[i][0], IDX[i][1], IDX[i][2])))

    # non-maximal supression
    if obj_class:
        sorted_indices = np.argsort(CONF[:, 0])[::-1]

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i][0].c == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i][0], boxes[index_j][0]) >= nms_threshold:
                        boxes[index_j][0].c = 0
        boxes = [box for box in boxes if box[0].c > obj_threshold]
    else:
        nonzero_cls = np.nonzero(CLASSES)[1]
        for cls in nonzero_cls:
            all_scores = CLASSES[:, cls]

            all_idx = np.where(all_scores > 0)[0]

            filtered_scores = CLASSES[all_idx, cls]

            sorted_indices = all_idx[np.argsort(filtered_scores)[::-1]]

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i][0].classes[cls] == 0:
                    continue
                else:
                    for j in range(i + 1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        if bbox_iou(boxes[index_i][0], boxes[index_j][0]) >= nms_threshold:
                            boxes[index_j][0].classes[cls] = 0
        boxes = [x for x in boxes if x[0].get_score() > obj_threshold]
    # remove the boxes which are less likely than a obj_threshold
    return boxes


def top_n_res(curr_res, n, reverse=False, wbb=False):
    label_dis = {}
    if wbb:
        for label, dis, bb, score, emb in curr_res:
            if label not in label_dis:
                label_dis[label] = [dis, bb, score, emb]
            elif reverse and label_dis[label][0] <= dis:
                label_dis[label] = [dis, bb, score, emb]
            elif not reverse and label_dis[label][0] >= dis:
                label_dis[label] = [dis, bb, score, emb]
        curr_res = [(k, v) for k, v in label_dis.items()]

        curr_res = sorted(curr_res, key=lambda x: x[1][0], reverse=reverse)
    else:
        for label, dis in curr_res:
            if label not in label_dis:
                label_dis[label] = dis
            elif reverse and label_dis[label] <= dis:
                label_dis[label] = dis
            elif not reverse and label_dis[label] >= dis:
                label_dis[label] = dis
        curr_res = [(k, v) for k, v in label_dis.items()]

        curr_res = sorted(curr_res, key=lambda x: x[1], reverse=reverse)

    return curr_res[:n]
