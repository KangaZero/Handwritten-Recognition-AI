import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Change to true if you want to train the model
train_new_model = False

if train_new_model:
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
    model.fit(x_train, y_train, epochs=3)
    
    # Evaluate the model with the test data
    loss, acc = model.evaluate(x_test, y_test)
    print('Loss: ', loss)
    print('acc: ', acc)

    model.save('handwritten.model')
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten.model')
    
    # Test with custom images and predict
    image_number = 1
# print(os.path.isfile(f"test_images/digit{image_number}.png"))
while os.path.isfile(f"test_images/digit{image_number}.png"):
    try: 
        # [:,:,0] is to get the first channel of the image (the grayscale image) since we do not care about colour
        img = cv2.imread(f"test_images/digit{image_number}.png")[:,:,0]
        # invert the image to fix it to the correct orientation
        img = cv2.resize(img, (28, 28))
        img = np.invert(np.array([img]))
    # Code below is not needed if all images are already 28x28
        # img = np.array(img).reshape(-1, 28, 28)
        # img = tf.keras.utils.normalize(img, axis=1)
        prediction = model.predict(img)
        print(f"Prediction for image {image_number}: {np.argmax(prediction)}")
        image_number += 1
    except:
        print(f"Error with image {image_number}")
        image_number += 1
    