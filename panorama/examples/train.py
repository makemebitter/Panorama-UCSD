from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pathmagic  # noqa
from panorama.config_gen import PanoramaConfig
from panorama.yoloembeddingnet.yoloembeddingnet import YoloEmbeddingNet
import argparse
import os
import random
import numpy as np
import tensorflow as tf


def set_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value

    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value

    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.set_random_seed(seed_value)


def main():
    config_gen = PanoramaConfig(args.config_savedir,
                                '',
                                args.train_annot_folder,
                                args.train_image_folder,
                                args.valid_annot_folder,
                                args.valid_image_folder,
                                '',
                                is_force=args.force_new_config
                                )
    config = config_gen.get_config()
    set_seeds(config['random_seed'])
    panoramaNet = YoloEmbeddingNet(config)

    panoramaNet.train(
        log_dir=args.log_dir,
        model_save_path=args.model_save_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        loss_weights=args.loss_weights,
        checkpoint=args.checkpoint,
        patience=args.patience,
        save_all=args.save_all,
        clipnorm=args.clipnorm
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_savedir", nargs='?',
        default='./faces_config.json',
    )
    parser.add_argument(
        "--checkpoint", type=str, nargs='?',
        default=''
    )
    parser.add_argument(
        "--train_annot_folder", type=str
    )
    parser.add_argument(
        "--train_image_folder", type=str
    )
    parser.add_argument(
        "--valid_annot_folder", type=str
    )
    parser.add_argument(
        "--log_dir", type=str
    )
    parser.add_argument(
        "--valid_image_folder", type=str
    )
    parser.add_argument(
        "--model_save_path", type=str
    )
    parser.add_argument(
        "--batch_size", type=int
    )
    parser.add_argument(
        "--learning_rate", type=float
    )
    parser.add_argument(
        "--force_new_config", action='store_true'
    )
    parser.add_argument(
        "--save_all", action='store_true'
    )
    parser.add_argument(
        "--patience", type=int, default=10
    )
    parser.add_argument(
        "--clipnorm", type=float, default=None
    )

    parser.add_argument(
        '--loss_weights', nargs='+', type=float, required=True)

    args = parser.parse_args()
    main()
