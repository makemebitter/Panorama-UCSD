#Copyright 2019 Yuhao Zhang
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
from sklearn.metrics import accuracy_score
import argparse
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from panorama.misctools.IO import load_obj
from panorama.misctools.IO import save_obj
from panorama.data.preprocess_cub import get_cub_summary


def main():

    LR = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', max_iter=100)

    index_fdir, \
        label_class, \
        index_bb, index_label, index_train_or_test = get_cub_summary(
            args.cub_dir)

    fea_train = load_obj(args.save_root, 'fea_train')
    label_train = load_obj(args.save_root, 'label_train')
    fea_val = load_obj(args.save_root, 'fea_val')
    label_val = load_obj(args.save_root, 'label_val')

    index_train = sorted(fea_train.keys())

    features_train = np.array([fea_train[key] for key in index_train])
    labels_train = np.array([label_train[key] for key in index_train])

    index_val = sorted(fea_val.keys())

    features_val = np.array([fea_val[key] for key in index_val])
    labels_val = np.array([label_val[key] for key in index_val])

    print(features_train.shape)
    print(labels_train.shape)
    print(features_val.shape)
    print(labels_val.shape)

    LR.fit(features_train, labels_train)
    labels_pred = LR.predict(features_train)
    accuracy = accuracy_score(labels_train, labels_pred)
    print("Training Accuracy:{}".format(accuracy))

    labels_pred = LR.predict(features_val)
    accuracy = accuracy_score(labels_val, labels_pred)
    print("Validation Accuracy:{}".format(accuracy))
    save_obj(LR, 'cvpr_lr', dir=args.lr_save_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--cub_dir", nargs='?',
        default='../../../../dataset/CUB_200_2011/CUB_200_2011',
        help="Directory for cub root"
    )

    parser.add_argument(
        "-s", "--save_root", nargs='?',
        default='../../../../produced/cvpr_extracted_features'
    )
    parser.add_argument(
        "-e", "--lr_save_root", nargs='?',
        default='../../../../trained_models'
    )
    args = parser.parse_args()
    main()
