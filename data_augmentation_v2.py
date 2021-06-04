# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:51:59 2021

@author: Raktim
"""
seed = 42
import tensorflow as tf
import os
import random
import numpy as np
from data_class import Data
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
#from models_new import *
#from sq_ex_block import *
#from models import *

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'C:/Users/rakti/Desktop/Test_Qupath/tiles/train/'
TEST_PATH = 'C:/Users/rakti/Desktop/Test_Qupath/tiles/test_all/'

data_obj= Data()
# You can provide your desired image size
# You can upscale or downscale

X_train, Y_train = data_obj.load_segmentation_data(TRAIN_PATH, 'tif', IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
X_test, Y_test = data_obj.load_segmentation_data(TEST_PATH, 'tif', IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Visualize image with corresponding mask
#data_obj.visualize(X_train,Y_train)

#Std Data Augmentation
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state = seed)



from keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

mask_data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     shear_range=0.5,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_data_generator.fit(X_train, augment=True, seed=seed)

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_data_generator.fit(Y_train, augment=True, seed=seed)




train_image = image_data_generator.flow(X_train, seed=seed)
val_image = image_data_generator.flow(X_val, seed=seed)

train_mask = mask_data_generator.flow(Y_train, seed=seed)
val_mask = mask_data_generator.flow(Y_val, seed=seed)


#i=0
#for batch in tqdm(image_data_generator.flow(X_train, batch_size=X_train.shape[0], 
#                                       save_to_dir='C:/Users/rakti/Desktop/U-Net/augmented/image', save_format='tif', seed=seed)):
#    i += 1
#    if i >= 3:
#        break  # otherwise the generator would loop indefinitely
#
#j=0
#for batch in tqdm(mask_data_generator.flow(Y_train, batch_size=Y_train.shape[0], 
#                                       save_to_dir='C:/Users/rakti/Desktop/U-Net/augmented/mask', save_format='tif', seed=seed)):
#    j += 1
#    if j >=3:
#        break  # otherwise the generator would loop indefinitely
        
def my_image_mask_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

train_generator = my_image_mask_generator(train_image, train_mask)

validation_generator = my_image_mask_generator(val_image, val_mask)

x = train_image.next()
y = train_mask.next()
for i in range(0,1):
    image = x[i]
    mask = y[i]
    plt.subplot(1,2,1)
    plt.imshow(image[:,:])
    plt.subplot(1,2,2)
    plt.imshow(mask[:,:,0], cmap='gray')
    plt.show()

    
    

from sklearn.model_selection import train_test_split
import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss, binary_crossentropy
from segmentation_models.metrics import iou_score, f1_score, recall, precision
from segmentation_models.utils import set_trainable
from segmentation_models import metrics, losses
from keras.models import load_model
from skimage import img_as_float, img_as_int

BACKBONE = 'seresnet50'

preprocess_input = sm.get_preprocessing(BACKBONE)

model_checkpoint = ModelCheckpoint('unet_v1_camelyon16.h5', monitor='loss', verbose=1, save_best_only=True)


#X_train = preprocess_input(X_train)

#X_train = preprocess_input(train_img)
#x_val = preprocess_input(_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), classes=1, encoder_freeze=True)
#model = sm.PSPNet(BACKBONE, encoder_weights='imagenet', input_shape=(240, 240, 3))


model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[metrics.iou_score])

print(model.summary())

batch_size = 50
steps_per_epoch = 3*(len(X_train))//batch_size


history = model.fit_generator(train_generator, validation_data=validation_generator, 
                              epochs=100, validation_steps=steps_per_epoch, steps_per_epoch=steps_per_epoch,
                              callbacks=[model_checkpoint])

#history = model.fit(train_generator, validation_steps=20, validation_data=validation_datagen, epochs=50, steps_per_epoch=100)


#steps_per_epoch=2000 // batch_size
#validation_steps=800 // batch_size

#model.load_weights('unet_seresnet50_camelyon16.h5')




objects={'binary_crossentropy_plus_dice_loss': bce_dice_loss, 'iou_score':iou_score}

model=load_model('unet_seresnet50_camelyon16.h5', custom_objects=objects)



#model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[iou_score])
model.summary()
#Evaluate the model


# evaluate model
_, acc = model.evaluate(X_test, Y_test)
print("Accuracy of Jacard Model is = ", (acc * 100.0), "%")



#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(Y_test, y_pred_thresholded)
union = np.logical_or(Y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)





ground_truth=Y_test.ravel()

predicted_score=(y_pred_thresholded.ravel())


from sklearn.metrics import roc_curve, auc 

fpr, tpr, _ = roc_curve(ground_truth, predicted_score)
roc_auc = auc(fpr,tpr)
print(roc_auc)


fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')
ax.legend(loc="lower right")






test_img_number = random.randint(0, len(X_test))
test_img = np.expand_dims(X_test[test_img_number],axis=0)
y_pred=model.predict(test_img)
prediction = (y_pred[0,:,:,0] > 0.5).astype(np.uint8)

ground_truth=Y_test[test_img_number]

test_image=X_test[test_img_number]



plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image)
plt.subplot(232)
plt.title('Ground Truth')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.savefig('ssss.png')
plt.show()




image_x = random.randint(0, len(X_train))
imshow(X_train[image_x])
plt.show()
imshow(Y_train[image_x], cmap='gray')
plt.show()
print(image_x)

image_x = random.randint(0, len(X_test))
imshow(X_test[image_x])
plt.show()
imshow(Y_test[image_x],cmap='gray')
plt.show()











