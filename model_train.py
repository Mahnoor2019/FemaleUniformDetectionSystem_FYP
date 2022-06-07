from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

size = 128

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory('data/train', target_size=(size, size),
                                                    batch_size=64, class_mode='binary')

validation_generator = validation_datagen.flow_from_directory('data/valid', target_size=(size, size),
                                                              batch_size=64, class_mode='binary')
# New Method
# history = model.fit(
#  train_generator,
# validation_data=validation_generator,
# epochs=20
# )

# Old Method

model.fit_generator(train_generator, epochs=5, steps_per_epoch=63,
                    validation_data=validation_generator, validation_steps=7, workers=4)

model.save('model.h5')
