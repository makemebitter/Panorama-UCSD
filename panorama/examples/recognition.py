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
from panorama.net.net import VerificationBase
from panorama.utils import bbox_iou
from panorama.utils import BoundBox
import argparse
import os
import random
import numpy as np
from panorama.misctools.utils import dir_and_class
from panorama.misctools.utils import PIL_to_cv2
from panorama.misctools.IO import save_obj
from PIL import Image


class Recognition(VerificationBase):
    def __init__(self,
                 panoramaNet,
                 img_folders,
                 ann_folders,
                 in_labels,
                 obj_thr,
                 nms_thr,
                 GT=False
                 ):
        super(Recognition, self).__init__(
            panoramaNet, img_folders, ann_folders, in_labels)
        self.obj_thr = obj_thr
        self.nms_thr = nms_thr
        self.GT = GT

    def preprocess_all(self):
        cached_data = {}
        for file_obj in self.file_list:
            filename = file_obj['filename']
            image_data = self.panoramaNet.load_from_disk(filename)
            image_h, image_w, _ = image_data.shape
            image_data = self.panoramaNet.preprocessed(image_data)
            cached_data[filename] = image_data, image_h, image_w
        return cached_data


    def detect_all(self, save_name, save_dir, sample_r=None):
        self.dir_class_pairs, self.class_dir_pairs = dir_and_class(
            self.file_list)
        if sample_r:
            self.class_dir_pairs = {k:random.sample(v, sample_r) for k, v in self.class_dir_pairs.items()}

        fdir_xml = {}
        for xml in self.file_list:
            fdir_xml[xml['filename']] = xml
        self.label_filename_embs = {}
        lcount = 0
        for label, dirs in self.class_dir_pairs.items():
            lcount += 1
            detected_list = []
            count = 0
            for fdir in dirs:
                count += 1
                print(
                    "label count:{}/{}, file count:{}/{}".format(
                        lcount,
                        len(self.class_dir_pairs),
                        count,
                        len(dirs)
                    )
                )
                xml = fdir_xml[fdir]
                objs = xml['object']
                img = Image.open(fdir).convert('RGB')
                img_cv2 = PIL_to_cv2(img)
                dur, emb_grid, netout, image_h, image_w = self.panoramaNet.get_raw(
                    img_cv2, 'all')
                depth_bests = [None] * len(emb_grid)
        #         loop of depths
                for i in range(len(emb_grid)):
                    best = [0, None, None, None]
                    raw = emb_grid[i], netout[i], image_h, image_w
                    out_boxes, \
                        out_scores, \
                        out_classes, \
                        embs = self.panoramaNet.decode_raw(
                            raw, self.obj_thr, self.nms_thr, True)
                    for out_bb, out_score, emb in zip(out_boxes,
                                                      out_scores, embs):
                        top, left, bottom, right = out_bb
                        out_BB = [left, top, right, bottom]
                        out_BB = BoundBox(*out_BB)
                        for obj in objs:
                            name = obj['name']
                            if name == label:
                                bb = BoundBox(
                                    obj['xmin'],
                                    obj['ymin'],
                                    obj['xmax'],
                                    obj['ymax']
                                )
                                iou = bbox_iou(out_BB, bb)
                                if iou > best[0]:
                                    best = [iou, out_score, emb, out_bb]
                    depth_bests[i] = best
                if all(depth_bests):
                    iou_score = np.mean([x[0] for x in depth_bests])
                    detected_list.append([fdir, iou_score, depth_bests])
            self.label_filename_embs[label] = detected_list
            save_obj(self.label_filename_embs, save_name, dir=save_dir)

    def run_helper(self, obj_thr, nms_thr, no_GT, neigh, k, filename_time, l_only=None, icaches=None, rec=False, rec_depth=None, cache_skip=1, cached_data=None):
        total_dur = [0] * (len(self.panoramaNet.depth_ls) + 1)
        solved_ins = {}
        count = 0
        GTCNNcount = 0
        total_faces = 0
        wrong = 0
        gt_time = 0
        total_embs_count = 0
        total_cache_hit_count = 0
        for file_obj in self.file_list:
            filename = file_obj['filename']
            labels = set([x['name'] for x in file_obj['object']])
            count += 1
            if icaches:
                durs, res_set, GT_invoked, solved_in, embs_count, cache_hit = \
                    self.panoramaNet.recognize(
                        filename, obj_thr, nms_thr, no_GT, neigh, k, l_only, icaches, cache_skip=cache_skip, cached_data=cached_data)
                total_embs_count += embs_count
                total_cache_hit_count += cache_hit
                cache_hit_rate = np.true_divide(
                    total_cache_hit_count, total_embs_count)
                print("Total embs:{}, Cache hit:{}, Hit rate:{}".format(
                    total_embs_count, total_cache_hit_count, cache_hit_rate))
            elif rec:
                image_data = self.panoramaNet.load_from_disk(filename)
                dur, curr_res, res_set = self.panoramaNet.predict(
                    image_data,
                    obj_thr,
                    nms_thr,
                    cascade_depth=rec_depth,
                    obj_class=False,
                    return_raw=False,
                    k=k)
                durs = [0] * (len(self.panoramaNet.depth_ls) + 1)
                durs[rec_depth] = dur
                GT_invoked = False
                solved_in = rec_depth
            else:
                durs, res_set, GT_invoked, solved_in = \
                    self.panoramaNet.recognize(
                        filename, obj_thr, nms_thr, no_GT, neigh, k, l_only, icaches, cached_data=cached_data)
            total_dur = [sum(x) for x in zip(total_dur, durs)]
            if GT_invoked and not no_GT:
                solved_in = 'GT'
            print (labels)
            print(res_set)
            total_faces += len(labels)
            if solved_in == 'GT':
                GTCNNcount += 1
                if not self.GT:
                    wrong += filename_time[filename][1]
                    gt_time += filename_time[filename][0]
                else:
                    wrong += 0
                    gt_time += filename_time[filename]
            else:
                if res_set:
                    wrong += len(labels - res_set)
                else:
                    wrong += len(labels)

            if solved_in in solved_ins:
                solved_ins[solved_in] += 1
            else:
                solved_ins[solved_in] = 1
            total_time_spent = sum(total_dur) + gt_time
            fps = count / total_time_spent
            acc = 1 - np.true_divide(wrong, total_faces)
            print(filename)
            print("Acc.:{}, FPS:{}".format(acc, fps))
        if icaches:
            return fps, acc, total_dur, solved_ins, count, total_embs_count, total_cache_hit_count, cache_hit_rate
        else:
            return fps, acc, total_dur, solved_ins, count


def get_all_test_dirs(root, which, splits=['test']):
    imgs = []
    anns = []
    for split in splits:
        all_imgs_dirs = []
        all_anns_dirs = []
        if which == 'out' or which == 'both':
            all_imgs_dirs.append(os.path.join(
                root, split, 'out', 'JPEGImages'))
            all_anns_dirs.append(os.path.join(
                root, split, 'out', 'Annotations'))
        if which == 'in' or which == 'both':
            all_imgs_dirs.append(os.path.join(root, split, 'in', 'JPEGImages'))
            all_anns_dirs.append(os.path.join(
                root, split, 'in', 'Annotations'))

        imgs += all_imgs_dirs
        anns += all_anns_dirs
    return imgs, anns


def main():

    all_imgs_dirs, all_anns_dirs = get_all_test_dirs(
        args.root, 'both', ['train', 'val'])

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
    random.seed(config['random_seed'])
    panoramaNet = PanoramaNet(config)
    panoramaNet.load_weights(args.model_save_path)
    rec = Recognition(panoramaNet, all_imgs_dirs, all_anns_dirs,
                      [], args.obj_thr, args.nms_thr)
    rec.detect_all(args.save_name, args.save_path)


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
    parser.add_argument(
        "--save_name", type=str
    )
    parser.add_argument(
        "--model_qualification_path", type=str
    )
    parser.add_argument(
        "--nms_thr", type=float, default=0.5
    )
    parser.add_argument(
        "--obj_thr", type=float, default=0.1
    )
    parser.add_argument(
        "--sample_r", type=int, default=None
    )

    args = parser.parse_args()
    main()
