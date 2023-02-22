import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values of the images
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the model
model = tf.keras.models.Sequential()

# Add the input layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Add the first hidden layer
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add the second hidden layer
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add the output layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
