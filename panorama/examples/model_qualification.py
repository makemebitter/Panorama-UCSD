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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import pathmagic  # noqa
from panorama.config_gen import PanoramaConfig
from panorama.net.net import PanoramaNet
import argparse
import os
from panorama.net.model_qulifier import ModelQualifier


def get_all_mq_dirs(root):
    all_imgs_dirs = []
    all_anns_dirs = []
    all_imgs_dirs.append(os.path.join(root, 'train', 'out', 'JPEGImages'))
    all_imgs_dirs.append(os.path.join(root, 'val', 'in', 'JPEGImages'))
    all_imgs_dirs.append(os.path.join(root, 'val', 'out', 'JPEGImages'))

    all_anns_dirs.append(os.path.join(root, 'train', 'out', 'Annotations'))
    all_anns_dirs.append(os.path.join(root, 'val', 'in', 'Annotations'))
    all_anns_dirs.append(os.path.join(root, 'val', 'out', 'Annotations'))

    return all_imgs_dirs, all_anns_dirs


def main():

    all_imgs_dirs, all_anns_dirs = get_all_mq_dirs(args.root)
    config_gen = PanoramaConfig(args.config_savedir,
                                '',
                                '',
                                '',
                                '',
                                '',
                                args.model_save_path,
                                is_force=False
                                )
    config = config_gen.get_config()
    panoramaNet = PanoramaNet(config)
    panoramaNet.load_weights(args.model_save_path)
    mq = ModelQualifier(panoramaNet,
                        all_imgs_dirs,
                        all_anns_dirs,
                        True,
                        args.save_path)
    mq.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_savedir", nargs='?',
        default='./faces_config.json',
    )
    parser.add_argument(
        "--root", type=str
    )
    parser.add_argument(
        "--model_save_path", type=str
    )
    parser.add_argument(
        "--save_path", type=str
    )

    args = parser.parse_args()
    main()
