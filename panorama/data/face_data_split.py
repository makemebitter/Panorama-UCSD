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
import glob
import numpy as np
from collections import defaultdict

import pathmagic  # noqa

from panorama.refmodels import RefFaceDetector
from panorama.refmodels import RefFaceExtractor
from panorama.misctools.utils import crop_all
from panorama.data.utilities import get_all_dirnames
from panorama.data.utilities import bb_apply_margin
from panorama.data.data_splitter import DataSplitter
import argparse
from PIL import Image
import hdbscan


class FaceSplitter(DataSplitter):
    def __init__(self,
                 frames_root,
                 destination,
                 extractor,
                 detector,
                 margin=44,
                 min_cluster_size=100
                 ):
        self.extractor = extractor
        self.detector = detector
        self.all_file_list = sorted(glob.glob(frames_root))
        self.destination = destination
        self.final_dir_names = get_all_dirnames(destination)

        self.margin = margin
        self.min_cluster_size = min_cluster_size
        self.splitter_name = 'face'

    def detect_all(self):
        count = 0
        self.filename_bb_emb = []
        self.filename_label = defaultdict(set)
        for image_path in self.all_file_list:
            image_data = self.detector.load_from_disk(image_path)
            out_boxes, out_scores, out_classes, duration = \
                self.detector.predict(
                    image_data)
            out_boxes = bb_apply_margin(out_boxes, self.margin, (1920, 1080))
            cropped = crop_all(Image.open(image_path), out_boxes)

            for cropped_image, out_box, out_score in \
                    zip(cropped, out_boxes, out_scores):
                if out_score > 0.99:
                    bb = np.array([int(x) for x in out_box], dtype=int)
                    count += 1
                    print ("Count:{}".format(count))
                    image_data_processed = self.extractor.load_from_pil(
                        cropped_image)
                    emb, dur = self.extractor.extract_features(
                        image_data_processed)
                    self.filename_bb_emb.append([image_path, bb, emb])
        self.generate_label()
        self.filename_bbx_label_raw = defaultdict(list)
        for filename, bb, label in self.filename_bb_label_list:
            self.filename_bbx_label_raw[filename].append([bb, label])

    def generate_label(self):
        list_embs = [x[2] for x in self.filename_bb_emb]
        model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        model.fit(list_embs)
        del list_embs
        labels = model.labels_
        effective_indices = np.nonzero(labels != -1)[0]
        self.effective_labels = set(labels)
        self.effective_labels.discard(-1)
        self.n_clusters = len(self.effective_labels)

        self.filename_bb_label_list = []
        for i in effective_indices:
            ele = self.filename_bb_emb[i]
            label = labels[i]
            filename_bb_label = [ele[0], ele[1], label]
            self.filename_bb_label_list.append(filename_bb_label)
        del self.filename_bb_emb


def main():

    face_detector = RefFaceDetector(args.mtcnn_weights)

    face_extractor = RefFaceExtractor(args.facenet_weights)

    data_splitter = FaceSplitter(
        args.frames_root,
        args.destination,
        face_extractor,
        face_detector
    )

    data_splitter.run()


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
        default='/datasets/home/28/228/yuz870/AdaFE/dataset/faces/dataset',
    )

    args = parser.parse_args()
    main()
