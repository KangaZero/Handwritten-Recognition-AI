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

# load the model

model = tf.keras.models.load_model('handwritten.model')

# test the model with the test data

loss, acc = model.evaluate(x_test, y_test)

print('Loss: ', loss)
print('acc: ', acc)

