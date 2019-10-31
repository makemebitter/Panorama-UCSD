from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
# import pafy #you need youtube-dl backend in order to use online streaming
from PIL import Image, ImageDraw, ImageFont
import random
import colorsys
import os
from collections import defaultdict
import itertools

package_directory = os.path.dirname(os.path.abspath(__file__))


def filter_boxes(out_boxes,
                 out_scores,
                 out_classes,
                 classes=[0],
                 width=30,
                 height=100):
    new_boxes = []
    new_scores = []
    new_classes = []
    for i in range(len(out_scores)):
        if (out_boxes[i][3] - out_boxes[i][1]) > width \
                and (out_boxes[i][2] - out_boxes[i][0] > height) \
                and (out_classes[i] in classes):
            new_boxes.append(out_boxes[i])
            new_scores.append(out_scores[i])
            new_classes.append(out_classes[i])
    return np.array(new_boxes), np.array(new_scores), np.array(new_classes)


class BoundingBoxDrawer(object):
    def __init__(self, LABELS=None, font_dir=None, colors=None, thickness=None,
                 FORMAT='PIL'):
        if font_dir:
            self.font_dir = font_dir
        else:
            self.font_dir = os.path.join(
                package_directory, 'font', 'FiraMono-Medium.otf')
        if colors:
            self.colors = colors
        else:
            self.colors = colors
        if FORMAT == 'cv2':
            self.draw = draw_boxes_cv2
        elif FORMAT == 'PIL':
            self.draw = self.__draw_boxes_PIL
            self.reset_color_map()
        self.color_label_map = {}
        self.LABELS = LABELS

    def reset_color_map(self, length=100):
        hsv_tuples = [(x / length, 1., 1.)
                      for x in range(length)]
    #     print (hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        # Shuffle colors to decorrelate adjacent classes.
        random.shuffle(colors)
        self.colors = colors
        return colors

    def save(self, fdir):
        self.image.save(fdir)

    def output(self,
               image,
               out_boxes,
               out_scores,
               out_classes,
               fdir,
               font=None,
               thickness=None
               ):
        image = self.__draw_boxes_PIL(image,
                                      out_boxes,
                                      out_scores,
                                      out_classes,
                                      font=None,
                                      thickness=None
                                      )
        image.save(fdir)

    def load_from_disk(self, img_path):
        return Image.open(img_path)

    def __draw_boxes_PIL(self,
                         image,
                         out_boxes,
                         out_scores,
                         out_classes,
                         use_plain_class_list=False,
                         font=None,
                         thickness=None,
                         obj_class=False,
                         cat="object",
                         colors=[]
                         ):
        is_cv2 = False
        colors_passed = True
        if type(image) is np.ndarray:
            image = cv2_to_PIL(image)
            is_cv2 = True
        if not thickness:
            thickness = (image.size[0] + image.size[1]) // 300
        if not font:
            font = ImageFont.truetype(font=self.font_dir, size=np.floor(
                3e-2 * image.size[1] + 20).astype('int32'))
        if not colors:
            colors = self.colors
            colors_passed = False
        for i, c in reversed(list(enumerate(out_classes))):
            if not use_plain_class_list:
                predicted_class = cat if obj_class else self.LABELS[c]
                color = colors[0] if obj_class else colors[c]
            else:
                predicted_class = c
                if colors_passed:
                    color = colors[i]
                else:
                    if predicted_class not in self.color_label_map:
                        self.color_label_map[predicted_class] = len(
                            self.color_label_map)
                    color = colors[self.color_label_map[predicted_class]]
            box = out_boxes[i]
            if out_scores.size:
                score = out_scores[i]
                label = '{} {:.2f}'.format(predicted_class, score)
            else:
                label = '{} '.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        if is_cv2:
            image = PIL_to_cv2(image)
        self.image = image
        return image


def cv2_to_PIL(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    return pil_img


def PIL_to_cv2(pil_img):
    cv2_img = np.array(pil_img)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    return cv2_img


def crop_all_del(image,
                 out_boxes,
                 out_classes,
                 out_scores,
                 labels=[],
                 allowed_labels=[],
                 cat="object"
                 ):
    if type(image) is str:
        image = Image.open(image)
    is_cv2 = False
    if type(image) is np.ndarray:
        image = cv2_to_PIL(image)
        is_cv2 = True
    cropped = []
    cropped_boxes = []
    cropped_classes = []
    cropped_scores = []
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = labels[c] if labels else cat
        box = out_boxes[i]
        score = out_scores[i]
        if not allowed_labels or predicted_class in allowed_labels:
            cropped_boxes.append(box)
            cropped_classes.append(c)
            cropped_scores.append(score)
            top, left, bottom, right = box
            cropped_single_image = image.crop([left, top, right, bottom])
            if is_cv2:
                cropped_single_image = PIL_to_cv2(cropped_single_image)
            cropped.append(cropped_single_image)
    return cropped, cropped_boxes, cropped_classes, cropped_scores


def crop_all(
        image,
        out_boxes):
    cropped = []
    for out_box in out_boxes:
        cropped.append(image.crop(out_box))
    return cropped


def draw_boxes_cv2(image, boxes, labels, allowed_labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        if labels[box.get_label()] in allowed_labels:
            xmin = int(box.xmin * image_w)
            ymin = int(box.ymin * image_h)
            xmax = int(box.xmax * image_w)
            ymax = int(box.ymax * image_h)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(image,
                        labels[box.get_label()] + ' ' + str(box.get_score()),
                        (xmin, ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h,
                        (0, 255, 0), 2)

    return image


# def get_video_url(url):
#     videoPafy = pafy.new(url)
#     best = videoPafy.getbest()
#     return best.url


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, image, clone
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            image = clone.copy()
            cv2.rectangle(image, refPt[0], (x, y), (0, 255, 0), 2)
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        # cv2.imshow("image", image)


def dir_and_class(valid_imgs):
    dir_class_pairs = {val['filename']: [x['name']
                                         for x in val['object']]
                       for val in valid_imgs}

    class_dir_pairs = defaultdict(list)

    for dir, classes in dir_class_pairs.items():
        for cls in classes:
            class_dir_pairs[cls].append(dir)
    return dir_class_pairs, class_dir_pairs


def sep_same_nonsame(valid_imgs, nsame=5000, nnonsame=5000):

    print ('Length of valid_imgs: {}'.format(len(valid_imgs)))

    dir_class_pairs, class_dir_pairs = dir_and_class(valid_imgs)

    dir_class_pairs_keys = dir_class_pairs.keys()
    class_dir_pairs_keys = class_dir_pairs.keys()
    random.shuffle(dir_class_pairs_keys)
    random.shuffle(class_dir_pairs_keys)

    total_list_same = []
    for cls in class_dir_pairs_keys:
        dirs = class_dir_pairs[cls]
        total_list_same += [x for x in itertools.combinations(dirs, 2)]
    same = random.sample(total_list_same, nsame)
    print ('Length of total_list_same: {}'.format(len(total_list_same)))
    del total_list_same
    total_list_nonsame = []
    for cls1 in class_dir_pairs_keys:
        dirs1 = class_dir_pairs[cls1]
        for cls2 in class_dir_pairs_keys:
            dirs2 = class_dir_pairs[cls2]
            if cls1 == cls2:
                continue
            else:
                comb_nonsame = list(itertools.product(dirs1, dirs2))
                for i, j in comb_nonsame:
                    u = set.intersection(
                        set(dir_class_pairs[i]), set(dir_class_pairs[j]))
                    if not u and set(dir_class_pairs[i]) \
                            and set(dir_class_pairs[j]):
                        total_list_nonsame.append((i, j))

    print ('Length of total_list_nonsame: {}'.format(len(total_list_nonsame)))
    nonsame = random.sample(total_list_nonsame, nnonsame)

    return same, nonsame
