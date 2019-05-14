from keras.datasets import cifar10
from keras import backend as K
from keras.applications.vgg16 import VGG16 as VG
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

import os
# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels...
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from items import item_dict

# define a VGG16 network

#Load image
K.set_image_dim_ordering("th")
#CHANGE IMGE NAME HERE
im = cv2.resize(cv2.imread('vice.jpg'), (224, 224)).astype(np.float32)
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

# Import model
#Weights https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
if os.path.exists('test_modelVG.h5'):
	model = load_model('test_modelVG.h5')
else:
	model = VG(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	model.save('test_modelVG.h5')

#VGG_16(weights_path='weights.h5')
optimizer = SGD()
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
out = model.predict(im)

index = np.argmax(out)

i = np.argsort(out)

print("Max Prediction: "+item_dict[int(index)])
print("Other predictions in order:")
ind = 1
for ind in range(5):
	name = item_dict[int(i[0][-ind-1])]
	print(str(ind)+". "+name)
