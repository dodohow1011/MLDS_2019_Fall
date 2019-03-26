import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
import sys
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def read():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
	# number = 10000
    np.random.shuffle(y_train[:30000])

	# x_train = x_train[0:number]
	# y_train = y_train[0:number]

    x_train = x_train.reshape((-1,28,28,1))/255
    x_test = x_test.reshape((-1,28,28,1))/255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = read()
model = Sequential()
model.add(Conv2D(256, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=20)

train = model.evaluate(x_train, y_train, batch_size=100)
test = model.evaluate(x_test, y_test, batch_size=64)

print ('Train Acc: ', train[1])
print ('Test Acc: ', test[1])
