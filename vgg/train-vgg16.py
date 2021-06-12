import numpy as np
import pandas as pd 
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (48,48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = 'categorical'
)

emotion_model = Sequential()

emotion_model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
emotion_model.add(Convolution2D(64, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(64, 3, 3, activation='relu'))
emotion_model.add(MaxPooling2D((2,2), strides=(2,2)))

emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(128, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(128, 3, 3, activation='relu'))
emotion_model.add(MaxPooling2D((2,2), strides=(2,2)))

emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(256, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(256, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(256, 3, 3, activation='relu'))
emotion_model.add(MaxPooling2D((2,2), strides=(2,2)))

emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(512, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(512, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(512, 3, 3, activation='relu'))
emotion_model.add(MaxPooling2D((2,2), strides=(2,2)))


emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(512, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(512, 3, 3, activation='relu'))
emotion_model.add(ZeroPadding2D((1,1)))
emotion_model.add(Convolution2D(512, 3, 3, activation='relu'))
emotion_model.add(MaxPooling2D((2,2), strides=(2,2)))

emotion_model.add(Flatten())
emotion_model.add(Dense(4096, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(4096, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch = 28709 // 64,
    epochs=75,
    validation_data = val_generator,
    validation_steps = 7178 // 64
)

emotion_model.save_weights('VGG16.h5')


