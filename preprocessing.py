import tensorflow as tf
import numpy as np
import cv2
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.utils import img_to_array
from PIL import Image

def preprocessing_img(GPU_n, img_path):
    with tf.device('device:GPU:{}'.format(GPU_n)):
        #image = cv2.imread(img_path)
        image = Image.fromarray(img_path)
        image = image.resize((512,256))
        image = img_to_array(image)
        image = tf.math.divide(image, 255.0)
        image = tf.cast(image, dtype=tf.float32)
        image = tf.expand_dims(image,0)

    return image

