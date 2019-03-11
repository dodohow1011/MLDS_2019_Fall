import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
import tensorflow as tf
def read():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
	
	# number = 10000
	
	# x_train = x_train[0:number]
	# y_train = y_train[0:number]

    x_train = x_train.reshape((-1,28,28,1))/255
    x_test = x_test.reshape((-1,28,28,1))/255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train,10)
    y_test = np_utils.to_categorical(y_test,10)

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = read()

    from deep_mnist import Deep
    # from shallow_mnist import Shallow
    # from two_layers_mnist import two_layers
    model = Deep()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])
    with tf.device('/gpu:0'):
        history = model.fit(x_train,y_train,batch_size=100,epochs=20)
    

if __name__ == '__main__':
    main()
