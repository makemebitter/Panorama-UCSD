# Copyright 2019 Yuhao Zhang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from lxml import objectify


class CONSTANTS():
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    ANNOTATIONS = 'Annotations'
    JPEGIMAGES = 'JPEGImages'
    PASCAL = 'pascal'
    IN_VOC = 'in'
    OUT_VOC = 'out'
    RANDOM_STATE = 42


def base_from_path(full_path):
    basename = os.path.basename(full_path)
    base = os.path.splitext(basename)[0]
    return base


def bb_apply_margin_helper(bb, margin, img_size):
    res_bb = [0, 0, 0, 0]
    width, height = img_size
    res_bb[0] = np.maximum(bb[0] - margin / 2, 0)
    res_bb[1] = np.maximum(bb[1] - margin / 2, 0)
    res_bb[2] = np.minimum(bb[2] + margin / 2, width)
    res_bb[3] = np.minimum(bb[3] + margin / 2, height)
    return res_bb


def bb_apply_margin(bbs, margin, img_size):
    new_bbs = []
    for bb in bbs:
        new_bbs.append(
            bb_apply_margin_helper(bb, margin, img_size)
        )
    return np.array(new_bbs)


def get_all_dirnames(root):
    pascal_root = os.path.join(root, CONSTANTS.PASCAL)
    final_dir_names = []
    dir_names = []
    for data_split in [CONSTANTS.TRAIN, CONSTANTS.VAL, CONSTANTS.TEST]:
        dir_names.append(os.path.join(pascal_root, data_split))
    for dirname in dir_names:
        for voc in [CONSTANTS.IN_VOC, CONSTANTS.OUT_VOC]:
            for pascal in [CONSTANTS.ANNOTATIONS, CONSTANTS.JPEGIMAGES]:
                dirname_voc_pascal = os.path.join(dirname, voc, pascal)
                final_dir_names.append(dirname_voc_pascal)
    return final_dir_names


def make_dirs(dir_list):
    for dirname in dir_list:
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def root(folder, filename, width, height, dataset_name):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
        E.folder(folder),
        E.filename(filename),
        E.source(
            E.database(dataset_name),
            E.annotation(dataset_name),
            E.image(dataset_name),
        ),
        E.size(
            E.width(width),
            E.height(height),
            E.depth(3),
        ),
        E.segmented(0)
    )


def obj_to_xml(bbx, label):
    E = objectify.ElementMaker(annotate=False)
    xmin, ymin, xmax, ymax = bbx
    return E.object(
        E.name(label),
        E.bndbox(
            E.xmin(xmin),
            E.ymin(ymin),
            E.xmax(xmax),
            E.ymax(ymax),
        ),
    )
