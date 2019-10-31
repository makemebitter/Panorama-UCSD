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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import cv2
import random

from sklearn.neighbors import KNeighborsClassifier
import pathmagic  # noqa
from panorama.config_gen import PanoramaConfig
from panorama.net.net import PanoramaNet
from panorama.net.net import K_nn
from panorama.misctools.utils import BoundingBoxDrawer


def select_bb(event, x, y, flags, param):
    global image, \
        clone, \
        out_boxes, \
        out_scores, out_classes, selected, box_index, touched, BBdrawer
    selected = False
    touched = False
    if event == cv2.EVENT_MOUSEMOVE:
        for i, box in enumerate(out_boxes):
            top, left, bottom, right = box
            if left < x < right and top < y < bottom:
                image = BBdrawer.draw(image,
                                      np.array([out_boxes[i]]),
                                      np.array([out_scores[i]]),
                                      np.array([out_classes[i]]),
                                      use_plain_class_list=True,
                                      colors=["#EF6F6C"]
                                      )
                touched = True
    elif event == cv2.EVENT_LBUTTONUP:
        for i, box in enumerate(out_boxes):
            top, left, bottom, right = box
            if left < x < right and top < y < bottom:
                box_index = i
                selected = True


class args:
    config_savedir = '../../trained_models/faces_config.json'
    k = 1
    model_save_path = \
        '../../trained_models/panorama_faces_original_loss_weights.h5'
    model_qualification_path = \
        '../../trained_models/panorama_faces_original_loss_weights.csv'
    nms_thr = 0.3
    obj_thr = 0.13
    ver_gamma = 0.9
    target = 0.9
    margin = 44
    source_type = 'file'
    source = "./trump_kim_clipped_scaled.mkv"


def main():
    global image, \
        clone, \
        out_boxes, \
        out_scores, out_classes, selected, box_index, touched, BBdrawer
    if args.source_type == 'file':
        capture_target = args.source
    elif args.source_type == 'camera':
        capture_target = args.source
    video_capture = cv2.VideoCapture(capture_target)

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

    neigh = KNeighborsClassifier(n_neighbors=1)
    CAT = "Face"
    BBdrawer = BoundingBoxDrawer()
    _, frame_previous = video_capture.read()
    skip_frame = 5
    LAYER = 2
    out_boxes, \
        out_scores, \
        out_classes, \
        embs = None, None, None, None

    neigh = K_nn(args.k, 'brute', [], [])

    skip_counter = 0
    while(video_capture.isOpened()):  # check!
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        image = frame.copy()
        clone = frame.copy()

        if ret == 0:
            break
        if(skip_counter == skip_frame):
            dur, emb_grid, netout, image_h, image_w = panoramaNet.get_raw(
                frame, LAYER)
            raw = emb_grid, netout, image_h, image_w
            out_boxes, \
                out_scores, \
                out_classes, \
                embs = panoramaNet.decode_raw(
                    raw, args.obj_thr, args.nms_thr, True)
            embs = np.asarray(embs)
            out_classes = out_classes.astype(str)
            out_classes = [CAT for x in range(len(out_classes))]
            if neigh.y_vals_train.size and out_boxes.size:
                dur, res_set, curr_res, curr_res_raw = neigh.predict(
                    embs,
                    args.k,
                    raw=True,
                    wbb=True,
                    out_boxes=out_boxes,
                    out_scores=out_scores)
                for i, res_tuple in enumerate(curr_res_raw):
                    label = res_tuple[0][0]
                    dis = res_tuple[1][0]
                    if dis < args.ver_gamma:
                        out_scores[i] = dis
                        out_classes[i] = label
            skip_counter = 0
        pred = [out_boxes, out_scores, out_classes]

        if not (any(x is None for x in pred)):
            frame = \
                BBdrawer.draw(frame, out_boxes, out_scores,
                              out_classes, use_plain_class_list=True)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1) & 0xFF
        skip_counter += 1
        if key == ord('s'):
            cv2.namedWindow("image")
            box_index = 0
            selected = False
            touched = False
            cv2.setMouseCallback("image", select_bb)
            while True:
                key = cv2.waitKey(1) & 0xFF
                # display the image and wait for a keypress
                if touched:
                    cv2.imshow("image", image)
                else:
                    image = frame.copy()
                    cv2.imshow("image", frame)

                if selected:
                    label = raw_input(
                        "Please enter the label for the selected object: ")
                    neigh.update([embs[box_index]], [str(label)])
                    selected = False
                # if the 'r' key is pressed, reset
                if key == ord("r"):
                    image = clone.copy()
                if key == ord("c"):
                    break
            cv2.destroyWindow("image")
        if key == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
