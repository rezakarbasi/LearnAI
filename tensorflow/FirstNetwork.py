import tensorflow as tf
import numpy as np

from tensorflow import keras
import matplotlib.pyplot as plt

import mnist as dataset


train_images = dataset.train_images()
train_labels = dataset.train_labels()

test_images = dataset.test_images()
test_labels = dataset.test_labels()

class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
