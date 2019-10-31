import numpy as np
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Conv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import keras.backend as K

import tensorflow as tf
from collections import defaultdict
import os, cv2,glob

import itertools
import random
from panorama.utils import WeightReader, decode_netout, draw_boxes,decode_netout_emb
import time



# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def generate_output_layer(GRID_H, GRID_W, BOX, CLASS,x,true_boxes,suffix='0'):
    features=x
    box_layer=Conv2D(BOX * (4 + 1 ), 
                    (1,1), strides=(1,1), 
                    padding='same', 
                    name='box_layer_'+suffix, 
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros')(features)
    
    box_layer = Reshape((GRID_H, GRID_W, BOX, (4 + 1 )))(box_layer)
    

    # total feature map
    emb_layer=Conv2D(BOX * (128 ), 
                    (1,1), strides=(1,1), 
                    padding='same', 
                    name='emb_layer_'+suffix, 
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros')(features)

    # proper shape
    emb_layer = Reshape((GRID_H, GRID_W, BOX, 128))(emb_layer)
    emb_layer=Lambda(lambda x: K.l2_normalize(x,axis=-1),name='l2_norm_layer_'+suffix)(emb_layer)
    class_layer=Conv3D(CLASS, 
                    (1,1,1), strides=(1,1,1), 
                    padding='same', 
                    name='class_layer_'+suffix, 
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros')(emb_layer)

    output=concatenate([box_layer,class_layer],axis=-1)             
    output=Lambda(lambda args: args[0])([output, true_boxes])
    return output

def yolo_tiny_cascade(GRID_H, GRID_W, BOX,CLASS,input_image_shape,true_boxes_shape,TINY_YOLO_BACKEND_PATH,input_tensor=None):
    print TINY_YOLO_BACKEND_PATH
    if input_tensor is not None:
        input_image = Input(tensor=input_tensor, shape=input_image_shape)
    else:
        input_image = Input(shape=input_image_shape)
    true_boxes  = Input(shape=true_boxes_shape)
    output=[]
    suffix_count=0
     # Layer 1
    x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2 - 5
    for i in range(0,4):
        x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(i+2))(x)
        x = LeakyReLU(alpha=0.1)(x)
        # if i == 3:
        #   output.append(generate_output_layer(GRID_H, GRID_W, BOX, x,'0'))
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    output.append(generate_output_layer(GRID_H, GRID_W, BOX, CLASS,x,true_boxes,suffix=str(suffix_count)))
    suffix_count+=1
    x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

    # Layer 7 - 8
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_7' , use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)
    output.append(generate_output_layer(GRID_H, GRID_W, BOX, CLASS,x,true_boxes,suffix=str(suffix_count)))
    suffix_count+=1
    x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)
    
    x = Conv2D(2048, (3,3), strides=(1,1), padding='same', name='conv_8' , use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    output.append(generate_output_layer(GRID_H, GRID_W, BOX, CLASS,x,true_boxes,suffix=str(suffix_count)))
    suffix_count+=1
            

            

        
#     model = Model(input_image, x)  
#     model.load_weights(TINY_YOLO_BACKEND_PATH)
#     model.summary()
#     grid_h, grid_w = model.get_output_shape_at(-1)[1:3]
    
    
    
    
#     features = model(input_image)  
 
    

    
    # make the object detection layer
#     output = Conv2D(BOX * (4 + 1 + CLASS), 
#                     (1,1), strides=(1,1), 
#                     padding='same', 
#                     name='DetectionLayer', 
#                     kernel_initializer='lecun_normal',
#                    bias_initializer='zeros')(features)
#     output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(output)
    
#     hack
    # output = Lambda(lambda args: args[0])([output, true_boxes])
    print output
    model = Model(inputs=[input_image, true_boxes], outputs=output)
        
        
        

    # initialize the weights of the detection layer
#     layer = model.layers[-4]
#     weights = layer.get_weights()

#     new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
#     new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

#     layer.set_weights([new_kernel, new_bias])

    # print a summary of the whole model
    model.summary()
    
    return model,input_image,true_boxes
def darknet(input_image_shape,true_boxes_shape,BOX,CLASS,GRID_H, GRID_W,input_tensor=None):
    if input_tensor is not None:
        input_image = Input(tensor=input_tensor, shape=input_image_shape)
    else:
        input_image = Input(shape=input_image_shape)
    true_boxes  = Input(shape=true_boxes_shape)

    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = Lambda(space_to_depth_x2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model 
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = Lambda(lambda args: args[0])([output, true_boxes])

    model = Model([input_image, true_boxes], output)
    return model

def custom_loss_deco(GRID_H,GRID_W,BATCH_SIZE,ANCHORS,BOX,COORD_SCALE,NO_OBJECT_SCALE,OBJECT_SCALE,CLASS_WEIGHTS,CLASS_SCALE,WARM_UP_BATCHES,true_boxes):
    def custom_loss(y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])

        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

        loss = loss_xy + loss_wh + loss_conf + loss_class

        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

        """
        Debugging code
        """    
        current_recall = nb_pred_box/(nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall) 

        loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
        loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
        loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
        loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
        loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
        loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)

        return loss
    return custom_loss


def get_embs(TRUE_BOX_BUFFER,ANCHORS,CLASS,dir,model,sub_model,obj_threshold=0.3,nms_threshold=0.5):
    image = cv2.imread(dir)
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
#     plt.figure(figsize=(10,10))

    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    netout = model.predict([input_image, dummy_array])
    intermediate_output = sub_model.predict([input_image, dummy_array])
    
    boxes = decode_netout_emb(netout[0], 
                          obj_threshold=obj_threshold,
                          nms_threshold=nms_threshold,
                          anchors=ANCHORS, 
                          nb_class=CLASS)
    
    return [intermediate_output[0][w][h][b] for box,(w,h,b) in boxes]

def get_embs_cascade(TRUE_BOX_BUFFER,ANCHORS,CLASS,dir,model,obj_threshold=0.3,nms_threshold=0.5,obj_class=False):
    image = cv2.imread(dir)
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    
    emb_grid,netout = model.predict([input_image, dummy_array])
    boxes = decode_netout_emb(netout[0], 
                          obj_threshold=obj_threshold,
                          nms_threshold=nms_threshold,
                          anchors=ANCHORS, 
                          nb_class=CLASS,
                          obj_class=obj_class)
    
    return [emb_grid[0][w][h][b] for box,(w,h,b) in boxes]
def get_embs_cascade_bchmk(TRUE_BOX_BUFFER,ANCHORS,CLASS,dir,model,obj_threshold=0.3,nms_threshold=0.5,obj_class=False):
    image = cv2.imread(dir)
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    yolofacenet_time=time.time()
    emb_grid,netout = model.predict([input_image, dummy_array])
    duration=time.time() - yolofacenet_time
    # print netout[0]
    boxes = decode_netout_emb(netout[0], 
                          obj_threshold=obj_threshold,
                          nms_threshold=nms_threshold,
                          anchors=ANCHORS, 
                          nb_class=CLASS,
                          obj_class=obj_class)
    # print boxes
    return [emb_grid[0][w][h][b] for box,(w,h,b) in boxes],duration
def get_acc(TRUE_BOX_BUFFER,ANCHORS,CLASS,data,model,intermediate_layer_model,verbose=False):
    dis=[]
    for i,j in data:
        if verbose:
            print i,j
        embsi=get_embs(TRUE_BOX_BUFFER,ANCHORS,CLASS,i,model,intermediate_layer_model)  
        embsj=get_embs(TRUE_BOX_BUFFER,ANCHORS,CLASS,j,model,intermediate_layer_model)
        dis_t=[]
        for embi in embsi:
            for embj in embsj:
                dis_t.append(np.linalg.norm(embi-embj))
        if dis_t:
            if verbose:
                print min(dis_t)
            dis.append(min(dis_t))
        else:
            if verbose:
                print 'no detection'
            dis.append(10)
    dis=np.array([dis])
    return dis

def get_acc_cascade(TRUE_BOX_BUFFER,ANCHORS,CLASS,data,model,obj_threshold=0.3,nms_threshold=0.5,verbose=False,obj_class=False):
    dis=[]
    for i,j in data:
        if verbose:
            print i,j
        embsi=get_embs_cascade(TRUE_BOX_BUFFER,ANCHORS,CLASS,i,model,obj_threshold=obj_threshold,nms_threshold=nms_threshold,obj_class=obj_class)  
        embsj=get_embs_cascade(TRUE_BOX_BUFFER,ANCHORS,CLASS,j,model,obj_threshold=obj_threshold,nms_threshold=nms_threshold,obj_class=obj_class)
        dis_t=[]
        # print embsi,embsj
        for embi in embsi:
            for embj in embsj:
                dis_t.append(np.linalg.norm(embi-embj))
        if dis_t:
            if verbose:
                print min(dis_t)
            dis.append(min(dis_t))
        else:
            if verbose:
                print 'no detection for one of {},{}'.format(i,j)
            dis.append(10)
    dis=np.array([dis])
    return dis
def get_dis_cascade(TRUE_BOX_BUFFER,ANCHORS,CLASS,data,model,verbose=False):
    dis=[]
    duration=[]
    for i,j in data:
        embsi=[]
        duri=0
        embsj=[]
        durj=0
        if verbose:
            print i,j
        for ii in i:
            # print ii
            embsii,durii=get_embs_cascade_bchmk(TRUE_BOX_BUFFER,ANCHORS,CLASS,ii,model)  
            embsi+=embsii
            duri+=durii
        for jj in j:
            embsjj,durjj=get_embs_cascade_bchmk(TRUE_BOX_BUFFER,ANCHORS,CLASS,jj,model)  
            embsj+=embsjj
            durj+=durjj 
        duration.append((duri,durj))
        dis_t=[]
        for embi in embsi:
            for embj in embsj:
                dis_t.append(np.linalg.norm(embi-embj))
        # print dis_t
        if dis_t:
            if verbose:
                print min(dis_t)
            dis.append(min(dis_t))
        else:
            print 'no detection for one of {},{}'.format(i,j)
            dis.append(10)
        # print dis
    dis=np.array([dis])

    duration=np.array([duration])
    return dis,duration


def sep_same_nonsame(valid_imgs,nsame=5000,nnonsame=5000):


    print 'Length of valid_imgs: {}'.format(len(valid_imgs))

    dir_class_pairs={val['filename']:[x['name'] for x in val['object']] for val in valid_imgs}

    class_dir_pairs=defaultdict(list)

    for dir, classes in dir_class_pairs.items():
        for cls in classes:
            class_dir_pairs[cls].append(dir)

    total_list_same=[]
    for cls, dirs in class_dir_pairs.items():
        total_list_same+=[x for x in itertools.combinations(dirs,2)]

    print 'Length of total_list_same: {}'.format(len(total_list_same))

    same=random.sample(total_list_same,nsame)

    total_list_nonsame=[]
    for cls1, dirs1 in class_dir_pairs.items():
        for cls2,dirs2 in class_dir_pairs.items():
            if cls1==cls2:
                continue
            else:
                total_list_nonsame+=list(itertools.product(dirs1,dirs2))
            
    print 'Length of total_list_nonsame: {}'.format(len(total_list_nonsame))
    total_list_nonsame_fil=[]
    for i,j in total_list_nonsame:
        u=set.intersection(set(dir_class_pairs[i]),set(dir_class_pairs[j]))
        # print i,j,u,set(dir_class_pairs[i]),set(dir_class_pairs[j])
        if not u and set(dir_class_pairs[i]) and set(dir_class_pairs[j]):
            total_list_nonsame_fil.append([i,j])
    # print total_list_nonsame_fil
    nonsame=random.sample(total_list_nonsame_fil,nnonsame)

    return same,nonsame
