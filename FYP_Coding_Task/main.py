from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Activation, Dropout
from keras.layers import Dense
import tensorflow as tf
from keras import callbacks
import os.path

from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# Step 1 - Convolution

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation("relu"))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='softmax'))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Compiling the CNN
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()


# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=12,
    class_mode='binary')
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=12,
    class_mode='binary')

# Part 3 - If the CNN weight model already exists make predictions
  if os.path.isfile("my_model.h5"):
    print("CNN Weight and models already exists, make the predictions")
# Part 3 - Else Load the data and store the CNN weight model
else:
    epochs = 5
    model.fit(
        training_set,
        validation_data=test_set,
        epochs=epochs
    )

model.save('my_model.h5')
