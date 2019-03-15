import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
import tensorflow as tf
import sys

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

    if len(sys.argv) != 2:
        print ('usage: python3 mnist_keras.py <option>')
        print ('options: deep2, deep4, deep6')

    x_train, y_train, x_test, y_test = read()

    if sys.argv[1] == 'deep6':
        from layers/six_layer import Six_layers
        model = Six_layers()
        history_filename = 'deep6_history.pickle'
        model_filename   = 'deep6_model.h5'
    elif sys.argv[1] == 'shallow' or sys.argv[1] == 'deep2':
        from layers/shallow_mnist import Shallow
        model = Shallow()
        history_filename = 'deep2_history.pickle'
        model_filename   = 'deep2_model.h5'
    elif sys.argv[1] == 'deep4':
        from layers/four_layers import Four_layers
        model = Four_layers()
        history_filename = 'deep4_history.pickle'
        model_filename   = 'deep4_model.h5'

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train,y_train,batch_size=100,epochs=20)
    model.save(model_filename)

#######################################################################
#                            save history                             #
#######################################################################
    import pickle
    with open(history_filename, 'wb') as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
