from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import numpy as np
from collections import defaultdict
from scipy.spatial import distance_matrix
from copy import deepcopy
import pandas as pd
from panorama.yoloembeddingnet.yoloembeddingnet import VerificationBase
from panorama.yoloembeddingnet.yoloembeddingnet import MQ_SCHEMA


class ModelQualifier(VerificationBase):
    def __init__(self,
                 panoramaNet,
                 img_folders,
                 ann_folders,
                 out_voc,
                 save_dir,
                 k=5
                 ):
        super(ModelQualifier, self).__init__(
            panoramaNet, img_folders, ann_folders)
        self.k = k
        self.save_dir = save_dir
        self.nms_threshold = 0.5
        self.thr_ls = [0.03, 0.1]
        self.gamma_ls = [0.2, 0.4, 0.6, 0.7, 0.8,
                         0.9, 1.0, 1.1, 1.2, 1.3]
        self.target_ls = [0.90, 0.95, 0.99]
        self.depth_ls = [0, 1, 2]
        self.out_voc = out_voc
        self.schema = MQ_SCHEMA
        self.sample = 5000
        self.rptimes = 3

    def run(self):
        self.df = []
        for i in range(self.rptimes):
            rseed = random.randint(0, 1000)
            random.seed(rseed)
            self.new_sample()
            for depth in self.depth_ls:
                thr_dis_same_nonsame = []
                for file_list in [self.same, self.nonsame]:
                    thr_dis = self.cal_dis(file_list, depth)
                    thr_dis_same_nonsame.append(thr_dis)
                for thr in self.thr_ls:
                    dis_same = thr_dis_same_nonsame[0][thr]
                    dis_nonsame = thr_dis_same_nonsame[1][thr]
                    same_tup = self.filter_dis(dis_same)
                    nonsame_tup = self.filter_dis(dis_nonsame)
                    for target in self.target_ls:
                        for gamma in self.gamma_ls:
                            acc_res = self.acc_helper(
                                target, gamma, same_tup, nonsame_tup)
                            (total_same, total_detected_same, _) = same_tup
                            (total_nonsame, total_detected_nonsame, _) = \
                                nonsame_tup
                            output = [thr,
                                      gamma,
                                      depth,
                                      target,
                                      rseed,
                                      total_same,
                                      total_nonsame,
                                      total_detected_same,
                                      total_detected_nonsame]
                            output += acc_res
                            self.df.append(output)
            self.ckpt_df = pd.DataFrame(self.df, columns=self.schema)
            self.ckpt_df.to_csv(self.save_dir, index=False)
        self.df = self.ckpt_df

    def cal_dis(self, file_list, cascade_depth):
        self.thr_dis = defaultdict(list)
        for files in file_list:
            thr_file_embs = defaultdict(list)
            print(files)
            for f in files:
                img_data = self.panoramaNet.load_from_disk(f)
                raw = self.panoramaNet.get_raw(img_data, cascade_depth)

                for obj_threshold in self.thr_ls:
                    _, _, _, embs = self.panoramaNet.decode_raw(
                        deepcopy(raw[1:]),
                        obj_threshold,
                        self.nms_threshold,
                        self.out_voc
                    )
                    embs = embs[:self.k]
                    thr_file_embs[obj_threshold].append(embs)
            for obj_threshold in self.thr_ls:
                file_embs = thr_file_embs[obj_threshold]
                if all(file_embs):
                    min_dis = np.min(distance_matrix(
                        file_embs[0], file_embs[1]))
                    self.thr_dis[obj_threshold].append(min_dis)
                else:
                    self.thr_dis[obj_threshold].append(-1)

        for obj_threshold in self.thr_ls:
            self.thr_dis[obj_threshold] = np.array(self.thr_dis[obj_threshold])
        return self.thr_dis

    def filter_dis(self, dis):
        total = dis.size
        dis = dis[dis != -1]
        total_detected = dis.size
        return total, total_detected, dis

    def acc_helper(self, target, thr, same_tup, nonsame_tup):
        (total_same, total_detected_same, dis_same) = same_tup
        (total_nonsame, total_detected_nonsame, dis_nonsame) = nonsame_tup
        ta_same = np.nonzero(dis_same <= thr)[0].size

        acc_sm = np.true_divide(ta_same, total_detected_same)
        print ('Accuracy for same: {}'.format(acc_sm))
        tr_nonsame = np.nonzero(dis_nonsame >= thr)[0].size

        acc_nsm = np.true_divide(tr_nonsame, total_detected_nonsame)
        print ('Accuracy for nonsame: {}'.format(acc_nsm))
        acc = np.true_divide(
            ta_same + tr_nonsame, total_detected_same + total_detected_nonsame)
        pre = np.true_divide(ta_same, ta_same +
                             total_detected_nonsame - tr_nonsame)
        rcl = np.true_divide(ta_same, ta_same +
                             total_detected_same - ta_same)

        TP = dis_same[dis_same <= thr]
        FP = dis_same[dis_same > thr]
        TN = dis_nonsame[dis_nonsame >= thr]
        FN = dis_nonsame[dis_nonsame < thr]
        target_error = (total_detected_nonsame +
                        total_detected_same) * (1 - target)
        total_error_samples = np.concatenate((FP, FN))
        if target_error >= total_error_samples.size:
            need_re_nm = 0
            low = 0
            high = 0
        else:
            per = (1 - target_error / total_error_samples.size) / 2
            total_dis = np.concatenate((dis_same, dis_nonsame))
            low = np.percentile(total_error_samples, (0.5 - per) * 100)
            high = np.percentile(
                total_error_samples, (0.5 + per) * 100)
            need_re_nm = np.where(
                (total_dis >= low) & (total_dis <= high))[0].size
        need_re_GT = total_nonsame + total_same - \
            total_detected_same - total_detected_nonsame
        output = [acc_sm, acc_nsm, acc, pre, rcl, ta_same,
                  tr_nonsame, low, high, need_re_nm, need_re_GT]
        return output
