# import cv2
import os
import dill
# import xml.etree.ElementTree as ET


# class ImagePreprocessed(object):
#     def __init__(self, model_name):
#         if model_name == 'yoloembeddingnet':
#             self.image = cv2.imread(dir)
#             image = cv2.imread(dir)
#             input_image = cv2.resize(image, (416, 416))
#             input_image = input_image / 255.
#             input_image = input_image[:, :, ::-1]
#             input_image = np.expand_dims(input_image, 0)
#         else:
#             raise NotImplementedError(
#                 "No model other than 'yoloembeddingnet' supported so far")
#         self.__name = os.path.splitext(os.path.basename(img_name))[0]

#     def __str__(self):
#         return self.__name

# class BaseBoundingBoxDrawer(object):
#     def __init__(self):
#          raise NotImplementedError("Implemented in subclasses")
#     def draw(self,image):


def load_obj(dir='.', name=None):
    if not name:
        with open(dir, 'rb') as f:
            res = dill.load(f)
    else:
        with open(os.path.join(dir, name + '.pkl'), 'rb') as f:
            res = dill.load(f)
    return res


def save_obj(obj, name, dir='.'):
    with open(os.path.join(dir, name + '.pkl'), 'wb+') as f:
        print os.path.join(dir, name + '.pkl')
        dill.dump(obj, f, dill.HIGHEST_PROTOCOL)

