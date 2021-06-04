#!/usr/bin/env python3 
__author__ = "Raktim Kumar Mondol"

"""
@author: Sreenivas Bhattiprolu
"""

import tensorflow as tf
import os
import random
import numpy as np
import cv2
 
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import img_as_float
from keras.models import load_model
seed = 42
np.random.seed = seed


# u-net model 
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
from models import *
from metrics import *
from losses import *

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = '/home/z5342745/data/camelyon16/train/'
TEST_PATH = '/home/z5342745/data/camelyon16/test/'

train_image_count = sum(len(files) for _, _, files in os.walk(r'/home/z5342745/data/camelyon16/train/'))
train_image_count=int(train_image_count/2)

test_image_count = sum(len(files) for _, _, files in os.walk(r'/home/z5342745/data/camelyon16/test/'))
test_image_count=int(test_image_count/2)



train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((train_image_count, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((train_image_count, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
i=0
j=0

#print('Resizing training images and masks')
#for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):  
#    path = TRAIN_PATH + id_
#    
#    for image_file in next(os.walk(path + '/image/'))[2]:
#        if (image_file.split('.')[1] == 'tif'):
#            image_ = imread(path + '/image/' + image_file)
#            image_=resize(image_, (IMG_WIDTH, IMG_HEIGHT), anti_aliasing=True)
#            X_train[i] = img_as_float(image_)
#            i=i+1
#        
#    for mask_file in next(os.walk(path + '/mask/'))[2]:
#        if (mask_file.split('.')[1] == 'tif'):
#            mask_ = imread(path + '/mask/' + mask_file)
#            mask_=resize(mask_, (IMG_WIDTH, IMG_HEIGHT), anti_aliasing=True)
#            grayscale = np.expand_dims(rgb2gray(mask_),axis=-1)
#            Y_train[j] = grayscale
#            j=j+1

# test images
k=0
f=0
X_test = np.zeros((test_image_count, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_test = np.zeros((test_image_count, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
sizes_test = []
print('Resizing test images') 

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    
    for image_file in next(os.walk(path + '/image/'))[2]:
        if (image_file.split('.')[1] == 'tif'):
            image_ = imread(path + '/image/' + image_file)
            image_=resize(image_, (IMG_WIDTH, IMG_HEIGHT), anti_aliasing=True)
            X_test[k] = img_as_float(image_)
            k=k+1
        
    for mask_file in next(os.walk(path + '/mask/'))[2]:
        if (mask_file.split('.')[1] == 'tif'):
            mask_ = imread(path + '/mask/' + mask_file)
            mask_=resize(mask_, (IMG_WIDTH, IMG_HEIGHT), anti_aliasing=True)
            grayscale = np.expand_dims(rgb2gray(mask_),axis=-1)
            Y_test[f] = grayscale
            f=f+1

print('Done!')


#image_x = random.randint(0, 5658)
#imshow(X_train[image_x])
#plt.show()
#imshow(np.squeeze(Y_train[image_x]))
#plt.show()
#print(image_x)
#
#image_x = random.randint(0, 3623)
#imshow(X_test[image_x])
#plt.show()
#imshow(Y_test[image_x],cmap='gray')
#plt.show()

#model = unet()

#model = U_Net_v2(input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

model = U_Net_v1(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#model = Nest_Net(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)

checkpoint = ModelCheckpoint('./unet_v1_2_camelyon16.h5', monitor='loss', verbose=0, save_best_only=True,save_weights_only=True)

earlystop = EarlyStopping(patience=3, monitor='loss')

#model.compile(optimizer = Adam(lr = 1e-3), loss = [jacard_coef_loss], metrics = [jacard_coef])
model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.compile(optimizer = Adam(lr = 1e-3), loss = [bce_dice_loss], metrics = [dice_coef])
#model.compile(optimizer = Adam(lr = 1e-3), loss = [jaccard_loss], metrics = [jaccard_coef])

model.summary()

#results = model.fit(X_train, Y_train, verbose=1, validation_split=0.25, batch_size=50, epochs=100, callbacks=[checkpoint,earlystop])

#model = load_model('unet_v1_camelyon16.hdf5')

model.load_weights('unet_v1_2_camelyon16.h5')

#Evaluate the model

# evaluate model
_, acc = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy of Dice Score is = ", (acc * 100.0), "%")


#IOU
y_pred=model.predict(X_test)
#y_pred_thresholded = y_pred > 0.5
#
#intersection = np.logical_and(Y_test, y_pred_thresholded)
#union = np.logical_or(Y_test, y_pred_thresholded)
#iou_score = np.sum(intersection) / np.sum(union)
#print("IoU socre is: ", iou_score)
#
#iou_np = iou_np(y_true=Y_test, y_pred=y_pred)
#print("IoU np socre is:", iou_np)

iou_thresholded_np=iou_thresholded_np(y_true=Y_test, y_pred=y_pred, threshold=0.5)
print("IoU thresholded_np socre is:", iou_thresholded_np)


###################################################
##Modelcheckpoint
#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_unet.h5', verbose=1, monitor='loss', save_best_only=True)
#
#callbacks = [
#        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#        tf.keras.callbacks.TensorBoard(log_dir='logs')]



