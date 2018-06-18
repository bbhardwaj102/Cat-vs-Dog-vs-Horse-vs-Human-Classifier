
import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.models import load_model


model = load_model('model.hdf5')
custom_vgg_model=load_model('custom_vgg_model.hdf5')
custom_vgg_model2=load_model('custom_vgg_model2.hdf5')

#img_path = 'elephant.jpeg'
while True:
    img_path = raw_input('Enter image path : ')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    print (x.shape)
    x = np.expand_dims(x, axis=0)
    print (x.shape)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    #pred1 = model.predict(x)
    pred2 = custom_vgg_model.predict(x)
    pred3 = custom_vgg_model2.predict(x)
    #print('Prediction 1 : ', pred1)
    print('Prediction 2 : ', pred2)
    print('Prediction 3 : ', pred3)
    #print('Predicted:', decode_predictions(pred1))
    c = raw_input('Do u want to continue (y or n) : ')
    if c == 'n':
        break