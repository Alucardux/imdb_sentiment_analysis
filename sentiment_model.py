# Imports relevant libraries
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import pandas as pd

# setting random seed
np.random.seed(18)

# Loading the preprocessed data from keras dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

# One-hot encoding the output into binary
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print('y train shape: {}'.format(y_train.shape))
print('y test shape: {}'.format(y_test.shape))


# Building the model architecture
model = Sequential()
model.add(Dense(400, activation='relu', input_dim=1000))
model.add(Dropout(.4))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

# Compiling the model using categorical_crossentropy loss, and adam optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Running and evaluating the model
hist = model.fit(x_train, y_train,
          batch_size=100,
          epochs=5,
          validation_data=(x_test, y_test),
          verbose=2)

# Model score in terms of Accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])

