from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

import os
from os.path import join

from PIL import Image, ImageFile

#PIL tuncated images error catch, keras have problem with it 
ImageFile.LOAD_TRUNCATED_IMAGES = True
#PILL DecompressionBomb warinng skip
Image.MAX_IMAGE_PIXELS = 99962095

path_to_folder = "/media/dmn/bbc6c909-5878-4a41-8590-51139f9491b2/artTainData/painter-by-numbers"
save_dir = "./model"
num_classes = 41 #number of art genres in training

model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(64,64,3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, kernel_size=3, border_mode="same"))
model.add(Activation("relu"))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
# model.add(LeakyReLU(alpha=0.2))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#model = VGG16()
train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        join(path_to_folder,"train"),
        target_size=(64, 64),
        batch_size=640,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        join(path_to_folder,'validation'),
        target_size=(64, 64),
        batch_size=640,
        class_mode='binary')

print(join(path_to_folder,'train'))
print(join(path_to_folder,'validation'))

model.fit_generator(
        train_generator,
steps_per_epoch=100,
validation_data=validation_generator,#可自定义
epochs=20,
# verbose=0,
validation_steps=320//64,
# epochs=100
# nb_val_samples=530
        )


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, "ImageGenreRecognition.h5")
model.save(model_path)
print('Saved trained model at %s ' % model_path)

scores = model.evaluate_generator(generator = validation_generator,steps=320//64,verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])