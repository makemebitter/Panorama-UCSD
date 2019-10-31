from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras import regularizers
import keras.backend as K
import tensorflow as tf
WEIGHT_DECAY = 0.00005

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)


def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)


def generate_output_layer(GRID_H,
                          GRID_W, BOX, CLASS, x, true_boxes, suffix='0'):
    features = x
    box_layer = Conv2D(BOX * (4 + 1),
                       (1, 1), strides=(1, 1),
                       padding='same',
                       name='box_layer_' + suffix,
                       kernel_initializer='lecun_normal',
                       bias_initializer='zeros',
                       kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                       )(features)

    box_layer = Reshape((GRID_H, GRID_W, BOX, (4 + 1)))(box_layer)

    # total feature map
    emb_layer = Conv2D(BOX * (128),
                       (1, 1), strides=(1, 1),
                       padding='same',
                       name='emb_layer_' + suffix,
                       kernel_initializer='lecun_normal',
                       bias_initializer='zeros',
                       kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                       )(features)

    # proper shape
    emb_layer = Reshape((GRID_H, GRID_W, BOX, 128))(emb_layer)
    emb_layer = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                       name='l2_norm_layer_' + suffix)(emb_layer)
    class_layer = Conv3D(CLASS,
                         (1, 1, 1), strides=(1, 1, 1),
                         padding='same',
                         name='class_layer_' + suffix,
                         kernel_initializer='lecun_normal',
                         bias_initializer='zeros',
                         kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                         )(emb_layer)

    output = concatenate([box_layer, class_layer], axis=-1)
    output = Lambda(lambda args: args[0])([output, true_boxes])
    return output


def yolo_tiny_cascade(GRID_H, GRID_W, BOX, CLASS, input_image_shape, true_boxes_shape, input_tensor=None):
    if input_tensor is not None:
        input_image = Input(tensor=input_tensor, shape=input_image_shape)
    else:
        input_image = Input(shape=input_image_shape)
    true_boxes = Input(shape=true_boxes_shape)
    output = []
    suffix_count = 0
    # Layer 1
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same',
               name='conv_1', use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
               )(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2 - 5
    for i in range(0, 4):
        x = Conv2D(32 * (2**i), (3, 3), strides=(1, 1), padding='same',
                   name='conv_' + str(i + 2), use_bias=False,
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                   )(x)
        x = BatchNormalization(name='norm_' + str(i + 2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        # if i == 3:
        #   output.append(generate_output_layer(GRID_H, GRID_W, BOX, x,'0'))
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
               name='conv_6', use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
               )(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    output.append(generate_output_layer(GRID_H, GRID_W, BOX,
                                        CLASS,
                                        x, true_boxes, suffix=str(suffix_count)))
    suffix_count += 1
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    # Layer 7 - 8
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same',
               name='conv_7', use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
               )(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)
    output.append(generate_output_layer(GRID_H,
                                        GRID_W, BOX,
                                        CLASS, x,
                                        true_boxes, suffix=str(suffix_count)))
    suffix_count += 1
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

    x = Conv2D(2048, (3, 3), strides=(1, 1), padding='same',
               name='conv_8', use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
               )(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    output.append(generate_output_layer(GRID_H, GRID_W, BOX,
                                        CLASS, x, true_boxes, 
                                        suffix=str(suffix_count)))
    suffix_count += 1
    # print (output)
    model = Model(inputs=[input_image, true_boxes], outputs=output)
    # model.summary()
    return model, input_image, true_boxes


def custom_loss_deco(GRID_H, GRID_W, BATCH_SIZE, ANCHORS, BOX, COORD_SCALE, NO_OBJECT_SCALE, OBJECT_SCALE, CLASS_WEIGHTS, CLASS_SCALE, WARM_UP_BATCHES, true_boxes):
    def custom_loss(y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(
            tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        # adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        # adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * \
            tf.reshape(ANCHORS, [1, 1, 1, BOX, 2])

        # adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4]) + 1e-6

        # adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        # adjust x and y
        # relative position to the containing cell
        true_box_xy = y_true[..., 0:2]

        # adjust w and h
        # number of cells accross, horizontally and vertically
        true_box_wh = y_true[..., 2:4]

        # adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        # adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        # coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

        # confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + \
            tf.to_float(best_ious < 0.6) * \
            (1 - y_true[..., 4]) * NO_OBJECT_SCALE

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

        # class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * \
            tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh +
                                                                tf.ones_like(
                                                                    true_box_wh) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2]) * no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy = tf.reduce_sum(
            tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(
            tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(
            true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(
            loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(tf.to_float(
            true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

        """
        Debugging code
        """
        current_recall = nb_pred_box / (nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall)

        loss = tf.Print(loss, [tf.zeros((1))],
                        message='Dummy Line \t', summarize=1000)
        loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
        loss = tf.Print(loss, [loss_conf],
                        message='Loss Conf \t', summarize=1000)
        loss = tf.Print(loss, [loss_class],
                        message='Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
        loss = tf.Print(
            loss, [current_recall], message='Curr. Recall Obj. Det. \t', summarize=1000)
        loss = tf.Print(loss, [total_recall / seen],
                        message='Avg. Recall Obj. Det. \t', summarize=1000)
        loss = tf.Print(loss, [tf.zeros((1))],
                        message='Dummy Line \t', summarize=1000)
        tf.summary.scalar('classification_loss', tensor=loss_class)
        tf.summary.scalar('avg_recall_obj_det', tensor=total_recall / seen)
        return loss
    return custom_loss
