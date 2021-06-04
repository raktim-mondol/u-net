#!/usr/bin/env python3  
"""
@author: Raktim Kumar Mondol 
"""

import tensorflow as tf
import os
import random
import numpy as np
import cv2

from data_class import Data
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from skimage.color import rgb2gray
from skimage import img_as_float
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import load_model
from matplotlib import pyplot as plt
# https://youtu.be/BNPW1mYbgS4
# u-net model 
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K

#from models import *

seed=42

TRAIN_PATH = 'C:/Users/rakti/Desktop/Test_Qupath/tiles/train/'
TEST_PATH = 'C:/Users/rakti/Desktop/Test_Qupath/tiles/test_all/'

data_obj= Data()


#PATH should contain TWO folder
#ONE is IMAGE folder and another one MASK folder
#This same structure has to be for both train and test image

X_train, Y_train = data_obj.load_segmentation_data(TRAIN_PATH, 'tif', 256, 256)

X_test, Y_test = data_obj.load_segmentation_data(TEST_PATH, 'tif', 256, 256)


#X_train = data_obj.load_classification_data(TRAIN_PATH, 'tif', 256, 256)




#p1='C:/D_DRIVE/UNSW/Experiment/qsub_test/data/colon/train/'
#X_train_colon, Y_train_colon = data_obj.load_segmentation_data(p1, 'png', 1024, 1024)


#p2='C:/D_DRIVE/UNSW/Experiment/Transfer_Learning/Dataset/Cats_and_Dogs/train/'
#X_train_cat_dog, Y_train_cat_dog = data_obj.load_classification_data(p2, 'jpg', 256, 256)


data_obj.label_check()
data_obj.visualize(X_train,Y_train)

image_x = random.randint(0, 20)
imshow(X_train_colon[image_x])
plt.show()
imshow(Y_train_colon[image_x],cmap='gray')
plt.show()

cp=np.expand_dims(Y_train_colon[image_x], axis=-1)
status = plt.imsave('ython_grey.png',cp, cmap='gray')
    
plt.imsave('Heatmap_Superimposed.png', cp, cmap='gray')
plt.show()
print(image_x)

image_x = random.randint(0, 3623)
imshow(X_test[image_x])
plt.show()
imshow(Y_test[image_x],cmap='gray')
plt.show()

#Std Data Augmentation

#This gives a binary mask rather than a mask with interpolated values. 
import numpy as np

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state = seed)


#
#from keras.preprocessing.image import ImageDataGenerator
#
#img_data_gen_args = dict(rotation_range=90,
#                     width_shift_range=0.3,
#                     height_shift_range=0.3,
#                     shear_range=0.5,
#                     zoom_range=0.3,
#                     horizontal_flip=True,
#                     vertical_flip=True,
#                     fill_mode='reflect')
#
#mask_data_gen_args = dict(rotation_range=90,
#                     width_shift_range=0.3,
#                     height_shift_range=0.3,
#                     shear_range=0.5,
#                     zoom_range=0.3,
#                     horizontal_flip=True,
#                     vertical_flip=True,
#                     fill_mode='reflect',
#                     preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 
#
#image_data_generator = ImageDataGenerator(**img_data_gen_args)
#image_data_generator.fit(X_train, augment=True, seed=seed)
#
#train_img = image_data_generator.flow(X_train, seed=seed)
#val_img = image_data_generator.flow(X_val, seed=seed)
#
#
#
#mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
#mask_data_generator.fit(Y_train, augment=True, seed=seed)
#
#train_mask = mask_data_generator.flow(Y_train, seed=seed)
#val_mask = mask_data_generator.flow(Y_val, seed=seed)



#for batch in image_data_generator.flow(X_train, batch_size=2000, 
#                                       save_to_dir='C:/Users/rakti/Desktop/U-Net/augmented/image', save_format='tif', seed=seed):
#    i += 1
#    if i > 2000:
#        break  # otherwise the generator would loop indefinitely
#
#for batch in mask_data_generator.flow(Y_train, batch_size=10, 
#                                       save_to_dir='C:/Users/rakti/Desktop/U-Net/augmented/mask', save_format='tif', seed=seed):
#    j += 1
#    if j > 1000:
#        break  # otherwise the generator would loop indefinitely
#        
#def my_image_mask_generator(image_generator, mask_generator):
#    train_generator = zip(image_generator, mask_generator)
#    for (img, mask) in train_generator:
#        yield (img, mask)
#
#train_generator = my_image_mask_generator(train_img, train_mask)
#
#validation_generator = my_image_mask_generator(val_img, val_mask)


#from matplotlib import pyplot as plt
#x = train_image_generator.next()
#y = train_mask_generator.next()
#for i in range(0,1):
#    image = x[i]
#    mask = y[i]
#    plt.subplot(1,2,1)
#    plt.imshow(image[:,:,0], cmap='gray')
#    plt.subplot(1,2,2)
#    plt.imshow(mask[:,:,0], cmap='gray')
#    plt.show()
#    
    
    

from sklearn.model_selection import train_test_split
import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.utils import set_trainable


BACKBONE = 'seresnet18'
preprocess_input = sm.get_preprocessing(BACKBONE)

#checkpoint = ModelCheckpoint('./unet_seresnet152_camelyon16.h5', monitor='loss', verbose=0, save_best_only=True,save_weights_only=True)

#earlystop = EarlyStopping(patience=3, monitor='loss')


#x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#X_train = preprocess_input(X_train)
#x_val = preprocess_input(_val)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(256, 256, 3), classes=1, encoder_freeze=True)

model.compile(optimizer='adam', loss=bce_jaccard_loss, metrics=[iou_score])

print(model.summary())


batch_size = 50
steps_per_epoch = 5*(len(X_train))//batch_size
#steps_per_epoch=2000 // batch_size
#validation_steps=800 // batch_size


history = model.fit_generator(train_generator, validation_data=validation_generator, 
                              epochs=100, validation_steps=steps_per_epoch, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint, earlystop])

#history = model.fit(train_generator, validation_steps=20, validation_data=validation_datagen, epochs=50, steps_per_epoch=100)


model.load_weights('unet_seresnet18_camelyon16.h5')


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


plt.imsave('Test_Image.png', X_test[test_img_number])

plt.imsave('Mask_Image.png', ground_truth[:,:,0], cmap='gray')

plt.imsave('Predicted_Image.png', prediction, cmap='gray')


img = cv2.imread('Test_Image.png', 1)
pred_mask = cv2.imread('Predicted_Image.png', 1)
#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
heatmap_img = cv2.applyColorMap(img, cv2.COLORMAP_JET )
fin = cv2.addWeighted(pred_mask, 0.6, heatmap_img, 0.4, 0)
plt.imsave('Heatmap_Superimposed.png', fin)
plt.imshow(fin)




loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()




#model.save('membrane.h5')
