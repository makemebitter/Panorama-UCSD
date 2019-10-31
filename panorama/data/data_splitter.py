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
from lxml import etree

from shutil import copy
from PIL import Image
import panorama.data.pathmagic  # noqa
from sklearn.model_selection import train_test_split
from panorama.data.utilities import CONSTANTS
from panorama.data.utilities import make_dirs
from panorama.data.utilities import base_from_path
from panorama.data.utilities import root
from panorama.data.utilities import obj_to_xml
from collections import defaultdict
from collections import OrderedDict


class DataSplitter():
    splitter_name = None
    final_dir_names = None
    destination = None

    def __init__(self):
        raise NotImplementedError

    def detect_all(self):
        raise NotImplementedError

    def run(self):
        make_dirs(self.final_dir_names)
        image = Image.open(self.all_file_list[0])
        self.width, self.height = image.size
        self.detect_all()
        self.total_detected_list = sorted(self.filename_bbx_label_raw.keys())
        self.generate_annotations()
        self.split()
        self.move_all()
        self.get_class_count()
        self.write_summary()

    def generate_annotations(self):
        self.filename_annotation = {}
        self.filename_label = defaultdict(set)
        for img_path in self.filename_bbx_label_raw.keys():
            basename = os.path.basename(img_path)
            annotation = root(
                CONSTANTS.PASCAL,
                basename, self.height, self.width, self.splitter_name)
            for bb, predicted_class \
                    in self.filename_bbx_label_raw[img_path]:
                annotation.append(obj_to_xml(bb, predicted_class))
                self.filename_label[img_path].add(predicted_class)
            self.filename_annotation[img_path] = annotation

    def split(self):
        self.in_voc, self.out_voc = self.in_out_voc_split()
        self.trian_file_list, self.test_file_list = train_test_split(
            self.total_detected_list,
            test_size=0.20, shuffle=False,
            random_state=CONSTANTS.RANDOM_STATE)
        self.trian_file_list, self.val_file_list = train_test_split(
            self.trian_file_list,
            test_size=0.25, shuffle=False,
            random_state=CONSTANTS.RANDOM_STATE)
        self.all_lists = self.in_out_data_split()

    def in_out_voc_split(self):
        self.total_vocabulary = set.union(*self.filename_label.values())
        self.total_vocabulary = sorted(list(self.total_vocabulary))
        in_voc, out_voc = train_test_split(
            self.total_vocabulary,
            test_size=0.20, shuffle=False, random_state=CONSTANTS.RANDOM_STATE)
        return in_voc, out_voc

    def move_all(self):
        for i, file_list in enumerate(self.all_lists):
            out_ann_dir = self.final_dir_names[2 * i]
            out_jpg_dir = self.final_dir_names[2 * i + 1]
            for filename in file_list:
                annotation = self.filename_annotation[filename]
                base = base_from_path(filename)
                annot_filename = os.extsep.join([base, 'xml'])
                ann_output_path = os.path.join(out_ann_dir, annot_filename)
                copy(filename, out_jpg_dir)
                etree.ElementTree(annotation).write(ann_output_path)

    def in_out_data_split_helper(self, filename_list):
        in_voc_filename_list = []
        out_voc_filename_list = []
        for filename in filename_list:
            labels = self.filename_label[filename]
            if labels.intersection(self.out_voc):
                out_voc_filename_list.append(filename)
            else:
                in_voc_filename_list.append(filename)
        return in_voc_filename_list, out_voc_filename_list

    def in_out_data_split(self):
        all_lists = []
        for filename_list in [
                self.trian_file_list, self.val_file_list, self.test_file_list]:
            in_voc_filename_list, \
                out_voc_filename_list = self.in_out_data_split_helper(
                    filename_list)
            all_lists.append(in_voc_filename_list)
            all_lists.append(out_voc_filename_list)
        return all_lists

    def get_class_count(self):
        self.class_count = {}
        for k, vs in self.filename_bbx_label_raw.items():
            for _, class_name in vs:
                if class_name in self.class_count:
                    self.class_count[class_name] += 1
                else:
                    self.class_count[class_name] = 1
        sorted_dict_tuples = [(x, self.class_count[x])
                              for x in sorted(self.class_count.keys())]
        self.class_count = OrderedDict(sorted_dict_tuples)

    def write_summary(self):
        summary_file = os.path.join(self.destination, 'summary.txt')
        total_count = len(self.total_detected_list)
        in_voc_train_count = len(self.all_lists[0])
        out_voc_train_count = len(self.all_lists[1])
        in_voc_val_count = len(self.all_lists[2])
        out_voc_val_count = len(self.all_lists[3])
        in_voc_test_count = len(self.all_lists[4])
        out_voc_test_count = len(self.all_lists[5])
        with open(summary_file, 'w+') as f:
            f.write("""
                Total voc:{self.total_vocabulary}\n
                In voc:{self.in_voc}\n
                Out voc:{self.out_voc}\n
                Total count:{total_count}\n
                In voc train count:{in_voc_train_count}\n
                Out voc train count:{out_voc_train_count}\n
                In voc val count:{in_voc_val_count}\n
                Out voc val count:{out_voc_val_count}\n
                In voc test count:{in_voc_test_count}\n
                Out voc test count:{out_voc_test_count}\n
                Each class count:{self.class_count}\n
                """.format(**locals()))
