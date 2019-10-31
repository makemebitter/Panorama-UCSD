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

from PIL import Image
import os
import sys
import cv2
import csv
import matplotlib.pyplot as plt
import random
from lxml import etree, objectify
import dill
from collections import OrderedDict
import numpy as np
import glob
import os
from PIL import Image
from collections import defaultdict
from shutil import copyfile
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('__file__'), '..')))
from panorama.misctools.IO import load_obj, parse_annotation, save_obj


def center_wh_to_xy_minmax(center_wh):
    x, y, w, h = center_wh
    return np.array([x - w // 2, y - h // 2, x + w // 2, y + h // 2])


def make_dirs(root_path, split_names=["train", "valid"]):
    save_dir = os.path.join(root_path, "pascal")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dst_dirs = {}
    ant_dirs = {}
    for split_name in split_names:
        split_dir = os.path.join(save_dir, split_name)
        dst_dir = os.path.join(split_dir, 'Images')
        ant_dir = os.path.join(split_dir, 'Annotations')
        dst_dirs[split_name] = dst_dir
        ant_dirs[split_name] = ant_dir
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        if not os.path.exists(ant_dir):
            os.makedirs(ant_dir)
    return save_dir, dst_dirs, ant_dirs


def root(folder, filename, width, height, name_of_dataset="name_of_dataset"):
    E = objectify.ElementMaker(annotate=False)
    return E.annotation(
        E.folder(folder),
        E.filename(filename),
        E.source(
            E.database(name_of_dataset),
            E.annotation(name_of_dataset),
            E.image(name_of_dataset),
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
        ))


def percentage_split(seq, percentages):
    assert sum(percentages) == 1.0
    prv = 0
    size = len(seq)
    cum_percentage = 0
    for p in percentages:
        cum_percentage += p
        nxt = int(cum_percentage * size)
        yield seq[prv:nxt]
        prv = nxt


root_path = "../../dataset/ytf/YouTubeFaces/frame_images_DB"
root_path = os.path.abspath(root_path)
pascal_path = "../../dataset/ytf/YouTubeFaces"
pascal_path = os.path.abspath(pascal_path)
split_names = ["train", "valid"]
save_dir, dst_dirs, ant_dirs = make_dirs(pascal_path, split_names)


def main():
    dirnames = sorted([x for x in os.listdir(root_path)
                       if os.path.isdir(os.path.join(root_path, x))])
    filename_bb_label = OrderedDict()
    for folder_dirname in dirnames:
        bb_filename = '.'.join(
            [os.path.join(root_path, folder_dirname), "labeled_faces", "txt"])
        with open(bb_filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            readCSV = [x for x in readCSV]
        train_split, valid_split = list(percentage_split(readCSV, [0.8, 0.2]))
        for name, split in [["train", train_split], ["valid", valid_split]]:
            for row in split:
                center_wh = np.array(row[2:6], dtype=int)
                filename = row[0].replace('\\', '/')
                if filename in filename_bb_label:
                    filename_bb_label[filename]['boundingboxs'].append(
                        (center_wh_to_xy_minmax(center_wh), folder_dirname))
                else:
                    filename_bb_label[filename] = OrderedDict()
                    filename_bb_label[filename]['boundingboxs'] = [
                        (center_wh_to_xy_minmax(center_wh), folder_dirname)]
                    filename_bb_label[filename]['split_name'] = name
    for filename, objs in filename_bb_label.items():
        filedir = os.path.join(root_path, filename)
        filename_out = filename.replace('/', '_')
        filename_out_base, filename_out_ext = os.path.splitext(filename_out)
        split_name = objs["split_name"]
        boundingboxs = objs["boundingboxs"]
        im = Image.open(filedir)
        width, height = im.size
        annotation = root('pascal', filename_out, height, width, "ytf")
        for bbx, label in boundingboxs:
            annotation.append(obj_to_xml(bbx, label))

        filedir_out = os.path.join(dst_dirs[split_name], filename_out)

        copyfile(filedir, filedir_out)
        etree.ElementTree(annotation).write(os.path.join(
            ant_dirs[split_name], filename_out_base + '.xml'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames_root", nargs='?',
        default='/datasets/home/28/228/yuz870/AdaFE/dataset/faces/raw/*.jpg',
    )
    parser.add_argument(
        "--mtcnn_weights", nargs='?',
        default='/datasets/home/28/228/yuz870/AdaFE/trained_models/align'
    )
    parser.add_argument(
        "--facenet_weights", nargs='?',
        default='/datasets/home/28/228/yuz870/AdaFE/trained_models/20170512-110547/20170512-110547.pb',
    )

    parser.add_argument(
        "--destination", nargs='?',
        default='/datasets/home/28/228/yuz870/AdaFE/dataset/faces/dataset_random_voc',
    )

    parser.add_argument(
        "--min_cluster_size", type=int, default=500
    )
    args = parser.parse_args()
    main()
