# -*- coding: utf-8 -*-
"""
Created on Mon May 24 13:16:41 2021

@author: Raktim
"""


from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Permute, multiply
from keras import backend as K
import tensorflow as tf

#tensor=tf.convert_to_tensor(
#    X_train, dtype=None, dtype_hint=None, name=None
#)


def squeeze_excite_block(tensor, nb_filter=32):
    init = tensor
    
    se= MaxPooling2D (pool_size=(2, 2), strides=(2,2), padding='same')(init)   
    
    se1=Conv2D(nb_filter, (1, 1), strides=(1,1), activation='relu', kernel_initializer = 'he_normal',  padding='same', kernel_regularizer=l2(1e-4))(se)
    se2=Conv2D(nb_filter, (3, 3), strides=(1,1), activation='sigmoid', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(se1)
    
    
    init= Conv2D(nb_filter, (3, 3), strides=(2,2), activation='relu', kernel_initializer = 'he_normal',  padding='same', kernel_regularizer=l2(1e-4))(init)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se2])
    return x


#pool1 = squeeze_excite_block(X_train)



