import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, ReLU, BatchNormalization, Dropout, Lambda

################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255.0)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path bin
    u6_bin = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6_bin = concatenate([u6_bin, c4])
    c6_bin = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6_bin)
    c6_bin = Dropout(0.2)(c6_bin)
    c6_bin = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_bin)
     
    u7_bin = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6_bin)
    u7_bin = concatenate([u7_bin, c3])
    c7_bin = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7_bin)
    c7_bin = Dropout(0.2)(c7_bin)
    c7_bin = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_bin)
     
    u8_bin = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7_bin)
    u8_bin = concatenate([u8_bin, c2])
    c8_bin = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8_bin)
    c8_bin = Dropout(0.1)(c8_bin)
    c8_bin = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_bin)
     
    u9_bin = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8_bin)
    u9_bin = concatenate([u9_bin, c1], axis=3)
    c9_bin = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9_bin)
    c9_bin = Dropout(0.1)(c9_bin)
    c9_bin = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_bin)
     
    bin_seg = Conv2D(1, (1, 1), activation='sigmoid', name='bin_seg')(c9_bin)
    
    #Expansive path inst
    u6_inst = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6_inst = concatenate([u6_inst, c4])
    c6_inst = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6_inst)
    c6_inst = Dropout(0.2)(c6_inst)
    c6_inst = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6_inst)
     
    u7_inst = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6_inst)
    u7_inst = concatenate([u7_inst, c3])
    c7_inst = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7_inst)
    c7_inst = Dropout(0.2)(c7_inst)
    c7_inst = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7_inst)
     
    u8_inst = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7_inst)
    u8_inst = concatenate([u8_inst, c2])
    c8_inst = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8_inst)
    c8_inst = Dropout(0.1)(c8_inst)
    c8_inst = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8_inst)
     
    u9_inst = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8_inst)
    u9_inst = concatenate([u9_inst, c1], axis=3)
    c9_inst = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9_inst)
    c9_inst = Dropout(0.1)(c9_inst)
    c9_inst = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9_inst)
     
    c9_inst = BatchNormalization()(c9_inst)
    c9_inst = ReLU()(c9_inst)
    inst_seg = Conv2D(4, (1, 1), activation='sigmoid', name='inst_seg')(c9_inst)
     
    model = Model(inputs=[inputs], outputs=[bin_seg, inst_seg])
    
    return model

def build_model(GPU_number, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, weight_data):
    with tf.device('device:GPU:{}'.format(GPU_number)):
        tf.get_logger().setLevel('ERROR')
        #tf.random.set_seed(40)
        tf.autograph.set_verbosity(10)
        model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        model.summary()
        model.load_weights(weight_data)
    return model

