from keras.datasets import mnist
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt

(tr_im, tr_label), (te_im, te_label) = mnist.load_data()

class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(tr_im[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[tr_label[i]])
# plt.show()

# preprocess the data

# first reshape the inputs to a 1D array
train_im = tr_im.reshape((60000, 28*28))
train_im = train_im.astype('float32')/255

test_im = te_im.reshape((10000, 28*28))
test_im = test_im.astype('float32')/255


# second : to use softmax classification we shold have n*10 array then categorize labels too
train_label = ku.to_categorical(tr_label)
test_label = ku.to_categorical(te_label)

# bulid the model from sequential
nn = models.Sequential()
# add first layer and configure hidden layer to 512 neurons and set it's activation function
nn.add(layers.Dense(512, activation='relu', input_shape=(
    28*28,), kernel_regularizer=regularizers.l2(0.01)))
# add final layer and set it to softmax layer
nn.add(layers.Dense(10, activation='softmax'))

# set optimizer optimizer parameters
nn.compile(optimizer='rmsprop', loss='categorical_crossentropy',
           metrics=['accuracy'])

# finally the fit function
history = nn.fit(train_im, train_label, validation_data=(
    test_im, test_label), epochs=50, batch_size=120)

# in the end evaluate the trained network by test dataset
print(nn.evaluate(test_im, test_label))

# show the results
train_loss = history.history['loss']
test_loss = history.history['val_loss']
x = list(range(1, len(test_loss) + 1))
plt.plot(x, test_loss, color='red', label='test loss')
plt.plot(x, train_loss, label='traning loss')
plt.show()
