# -*- coding: utf-8 -*-
"""1. RGB-ImagesCIFAR10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AQsenK9C5GyM70W3zpbsa_SavbqRaXMS
"""

import tensorflow as tf

# Display the version
print(tf.__version__)

# other imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# visualize data by plotting images
fig, ax = plt.subplots(5, 5)
k = 0

for i in range(5):
	for j in range(5):
		ax[i][j].imshow(x_train[k], aspect='auto')
		k += 1

plt.show()

# Convert the images to RGB
x_train_rgb = np.repeat(x_train, 3, axis=-1)
x_test_rgb = np.repeat(x_test, 3, axis=-1)

print("Original shape:", x_train.shape)
print("RGB shape:", x_train_rgb.shape)

# Assuming you have an image array called "image"
num_channels = x_train_rgb.shape[-1]
num_channels2 = x_test_rgb.shape[-1]

print("Number of channels:", num_channels)
print("Number of channels:", num_channels2)

# Convert the images to float32
# Assuming you have RGB images in x_train_rgb and x_test_rgb
x_train_float32 = x_train_rgb.astype('float32')
x_test_float32 = x_test_rgb.astype('float32')

print("Original data type:", x_train_rgb.dtype)
print("Float32 data type:", x_train_float32.dtype)


# Normalize the pixel values to the range [0, 1]
# Assuming you have RGB images in x_train_float32 and x_test_float32

# Normalize the pixel values to the range [0, 1]
x_train_normalized = x_train_float32 / 255.0
x_test_normalized = x_test_float32 / 255.0

print("Original pixel value range:", x_train_float32.min(), x_train_float32.max())
print("Normalized pixel value range:", x_train_normalized.min(), x_train_normalized.max())

# Extract the red channel from each image
# Assuming you have RGB images in x_train_normalized and x_test_normalized

# Extract the red channel from each image
x_train_red = x_train_normalized[..., 0]
x_test_red = x_test_normalized[..., 0]

print("Original shape:", x_train_normalized.shape)
print("Red channel shape:", x_train_red.shape)



# Assuming you have red channel arrays in x_train_red and x_test_red

# Reshape the red channel arrays
x_train_red_reshaped = np.expand_dims(x_train_red, axis=-1)
x_test_red_reshaped = np.expand_dims(x_test_red, axis=-1)

print("Original shape:", x_train_red.shape)
print("Reshaped shape:", x_train_red_reshaped.shape)

# Assuming you have RGB images in x_train_normalized and x_test_normalized

# Extract the green channel from each image
x_train_green = x_train_normalized[..., 1]
x_test_green = x_test_normalized[..., 1]

print("Original shape:", x_train_normalized.shape)
print("Green channel shape:", x_train_green.shape)


# Assuming you have green channel arrays in x_train_green and x_test_green

# Reshape the green channel arrays
x_train_green_reshaped = np.expand_dims(x_train_green, axis=-1)
x_test_green_reshaped = np.expand_dims(x_test_green, axis=-1)

print("Original shape:", x_train_green.shape)
print("Reshaped shape:", x_train_green_reshaped.shape)

# Assuming you have RGB images in x_train_normalized and x_test_normalized

# Extract the blue channel from each image
x_train_blue = x_train_normalized[..., 2]
x_test_blue = x_test_normalized[..., 2]

print("Original shape:", x_train_normalized.shape)
print("Blue channel shape:", x_train_blue.shape)


# Assuming you have blue channel arrays in x_train_blue and x_test_blue

# Reshape the blue channel arrays
x_train_blue_reshaped = np.expand_dims(x_train_blue, axis=-1)
x_test_blue_reshaped = np.expand_dims(x_test_blue, axis=-1)

print("Original shape:", x_train_blue.shape)
print("Reshaped shape:", x_train_blue_reshaped.shape)

from sklearn.model_selection import StratifiedShuffleSplit

s1=StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1/6)
train_index1, valid_index1 = next(s1.split(x_train_red_reshaped, y_train))
x_valid1, y_valid1 = x_train_red_reshaped[valid_index1], y_train[valid_index1]
x_train1, y_train1 = x_train_red_reshaped[train_index1], y_train[train_index1]

print(x_train1.shape, x_valid1.shape, x_test_red_reshaped.shape)

s2=StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1/6)
train_index2, valid_index2 = next(s2.split(x_train_green_reshaped, y_train))
x_valid2, y_valid2 = x_train_green_reshaped[valid_index2], y_train[valid_index2]
x_train2, y_train2 = x_train_green_reshaped[train_index2], y_train[train_index2]

print(x_train2.shape, x_valid2.shape, x_test_green_reshaped.shape)

s3=StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1/6)
train_index3, valid_index3 = next(s3.split(x_train_blue_reshaped, y_train))
x_valid3, y_valid3 = x_train_blue_reshaped[valid_index3], y_train[valid_index3]
x_train3, y_train3 = x_train_blue_reshaped[train_index3], y_train[train_index3]

print(x_train3.shape, x_valid3.shape, x_test_blue_reshaped.shape)

# number of classes
K = len(set(y_train))

# calculate total number of classes
# for output layer
print("number of classes:", K)

# Build the model using the functional API
# input layer
i = Input(shape=x_train_red_reshaped[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# last hidden layer i.e.. output layer
x = Dense(K, activation='softmax')(x)

model1 = Model(i, x)

# model description
model1.summary()

# Compile
model.compile(optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy'])

# Fit
r = model.fit(
x_train, y_train, validation_data=(x_test, y_test), epochs=50)