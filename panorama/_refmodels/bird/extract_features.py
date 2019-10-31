from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse


import pathmagic  # noqa
from panorama.data.preprocess_cub import get_cub_summary
from panorama.misctools.IO import save_obj
from panorama.refmodels import RefBirdClassifier


def main():
    image_dir = os.path.join(args.cub_dir, args.raw_img_dir)

    index_fdir, \
        label_class, \
        index_bb, index_label, index_train_or_test = get_cub_summary(
            args.cub_dir)

    # Feature extraction.
    fea_train = {}
    label_train = {}
    fea_val = {}
    label_val = {}
    ref_model = RefBirdClassifier(cnn_model_path=args.checkpoints_path)
    count = 0
    total_dur = 0
    for index in sorted(index_fdir.keys()):
        label = index_label[index]
        fdir = index_fdir[index]
        file_fulldir = os.path.join(image_dir, fdir)
        image_data = ref_model.load_from_disk(file_fulldir)
        fea, curr_dur = ref_model.extract_features(image_data)
        total_dur += curr_dur
        count += 1
        print("Index:{}, Curr FPS:{}, Count:{}".format(
            index, count / total_dur, count))
        if index_train_or_test[index] == 1:
            fea_train[index] = fea
            label_train[index] = label
        elif index_train_or_test[index] == 0:
            fea_val[index] = fea
            label_val[index] = label

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    save_obj(fea_train, 'fea_train', dir=args.save_root)
    save_obj(label_train, 'label_train', dir=args.save_root)
    save_obj(fea_val, 'fea_val', dir=args.save_root)
    save_obj(label_val, 'label_val', dir=args.save_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--cub_dir", nargs='?',
        default='../../../../dataset/CUB_200_2011/CUB_200_2011',
        help="Directory for cub root"
    )
    parser.add_argument(
        "-v", "--raw_img_dir", nargs='?',
        default='images'
    )
    parser.add_argument(
        "-s", "--save_root", nargs='?',
        default='../../../../produced/cvpr_extracted_features'
    )
    parser.add_argument(
        "-e", "--checkpoints_path", nargs='?',
        default='../../../../trained_models/inception_v3_iNat_299.ckpt',
        help="Path for the embedding model"
    )
    args = parser.parse_args()
    main()
