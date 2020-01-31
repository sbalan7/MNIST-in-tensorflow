# -*- coding: utf-8 -*-
"""
Using tensorflow on the mnist fashion dataset

@author: Nabla
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Getting and loading the dataset
fashion = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion.load_data()

# The labels for the dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Normalising the image data to the interval [0, 1]
train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0

"""
# To view the dataset
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_imgs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

# Making the model layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_imgs, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_imgs,  test_labels, verbose=2)


# Predicting data
predictions = model.predict(test_imgs)
print(class_names[np.argmax(predictions[0])])
print(predictions[0])
plt.imshow(test_imgs[0])
