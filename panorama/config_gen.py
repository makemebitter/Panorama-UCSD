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

from gen_anchors import get_anchors, sorted_anchors
from preprocessing import parse_annotation
import json
import os

from collections import namedtuple
import random

Dataset = namedtuple('Dataset',
                     ['random_seed',
                      'IMAGE_H',
                      'IMAGE_W',
                      'GRID_H',
                      'GRID_W',
                      'BOX',
                      'NO_OBJECT_SCALE',
                      'OBJECT_SCALE',
                      'COORD_SCALE',
                      'CLASS_SCALE',
                      'WARM_UP_BATCHES',
                      'TRUE_BOX_BUFFER',
                      'wt_path',
                      'train_image_folder',
                      'train_annot_folder',
                      'valid_image_folder',
                      'valid_annot_folder',
                      'model_save_path',
                      'LABELS',
                      'CLASS',
                      'CLASS_WEIGHTS',
                      'ANCHORS'])
DEFAULT_CONFIG = Dataset(42,
                         416,
                         416,
                         13,
                         13,
                         5,
                         1.0,
                         5.0,
                         1.0,
                         1.0,
                         0,
                         50,
                         '',
                         '',
                         '',
                         '',
                         '',
                         '',
                         [],
                         0,
                         [],
                         []
                         )._asdict()


class PanoramaConfig():
    def __init__(self,
                 config_savedir,
                 wt_path,
                 train_annot_folder,
                 train_image_folder,
                 valid_annot_folder,
                 valid_image_folder,
                 model_save_path,
                 is_force=False
                 ):
        self.config_savedir = config_savedir
        self.is_force = is_force
        self.wt_path = wt_path
        self.train_annot_folder = train_annot_folder
        self.train_image_folder = train_image_folder
        self.valid_annot_folder = valid_annot_folder
        self.valid_image_folder = valid_image_folder
        self.model_save_path = model_save_path
        self.config = DEFAULT_CONFIG

    def write_all_missing_fields(self):
        self.config['wt_path'] = self.wt_path
        self.config['train_annot_folder'] = self.train_annot_folder
        self.config['train_image_folder'] = self.train_image_folder
        self.config['valid_annot_folder'] = self.valid_annot_folder
        self.config['valid_image_folder'] = self.valid_image_folder
        self.config['model_save_path'] = self.model_save_path
        _, seen_labels = parse_annotation(
            self.train_annot_folder,
            self.train_image_folder,
            labels=[],
            onlyInLabels=False
        )
        LABELS = sorted(seen_labels.keys())
        self.config['LABELS'] = LABELS
        self.config['CLASS'] = len(LABELS)
        self.config['CLASS_WEIGHTS'] = [1.0] * len(LABELS)
        random.seed(self.config['random_seed'])
        anchors = sorted_anchors(get_anchors(self.config))
        self.config['ANCHORS'] = anchors

    def get_config(self):

        if self.is_force:
            self.write_all_missing_fields()
            with open(self.config_savedir, 'w') as fp:
                json.dump(self.config, fp, sort_keys=True, indent=4)
        else:
            if os.path.isfile(self.config_savedir):
                with open(self.config_savedir, 'r') as fp:
                    self.config = json.load(fp)
            else:
                raise ValueError('Config file not found.')

        return self.config
