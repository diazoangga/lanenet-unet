from numpy.lib.function_base import insert
import tensorflow as tf
import numpy as np
import cv2
import warnings
import numpy as np
import time

def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def inference(GPU_number, model, imageRGB):
    with tf.device('device:GPU:{}'.format(GPU_number)):
        tf.get_logger().setLevel('ERROR')
        #str = time.time()
        bin_pred, inst_pred = model.predict(imageRGB)
        #end = time.time()
        #print(str - end)
        bin_pred = tf.squeeze(bin_pred, axis=0)
        inst = tf.squeeze(inst_pred, axis=0)
        threshold = tf.constant(0.4, shape=(256,512,3))
        bin = tf.math.greater(bin_pred, threshold)
        bin = tf.cast(bin, dtype=tf.uint8)
        #bin = tf.image.rgb_to_grayscale(bin)
        #bin = (bin_pred[0,:,:,0] > 0.3).astype(np.uint8)
        #inst = inst_pred[0,:,:,:]
        #bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
        for i in range(4):
            inst_pred[0,:,:,i] = minmax_scale(np.array(inst_pred[0][:, :, i]))
            embedding_image = np.array(inst_pred[0], np.uint8)
        result = tf.math.multiply(bin, tf.convert_to_tensor(embedding_image[:,:,(2,1,0)]))
        result = tf.cast(result, dtype=tf.float32)
        in_img = tf.math.multiply(255.0, tf.squeeze(imageRGB, axis=0))
        in_img = tf.cast(in_img, dtype=tf.float32)
        
        merge_img = tf.math.add(tf.math.multiply(in_img, 0.6), tf.math.multiply(result, 0.4))
        merge_img = tf.cast(merge_img,dtype=tf.uint8)
        #result = tf.cast(result, dtype=tf.uint8)
        # print((np.array(imageRGB)[0]*255).astype(int).shape)
        # print(np.array(result).shape)
        # result = cv2.addWeighted((np.array(imageRGB)[0]*255).astype(int), 0.5, np.array(result), 0.5, 0.0)
        # result = result.astype(np.uint8)
        #cv2.imshow('restult', result)
    
    return np.array(merge_img)

