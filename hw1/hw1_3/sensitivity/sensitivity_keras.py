from keras import optimizers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score
import sys

def read():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1,28,28,1)/255
    x_test = x_test.reshape(-1,28,28,1)/255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)

    return x_train, y_train, x_test, y_test

def deep4():
    model = Sequential()
    model.add(Conv2D(144, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(144, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(144, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(144, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.summary()
    return model


def training(number):
    x_train, y_train, x_test, y_test = read()
    conv2d_1_input = tf.placeholder(tf.float32,shape=(None,28,28,1))
    model = deep4()
    with tf.Session() as sess:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train,y_train,batch_size=number,epochs=30)
#    y_train = tf.convert_to_tensor(y_train)
#    model.outputs = tf.convert_to_tensor(model.outputs)
#    loss = tf.keras.losses.categorical_crossentropy(y_train, model.outputs)
#        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
#        y_output = model.apply(x_test)
#        y_test = tf.convert_to_tensor(y_test)
        y_output = model.apply(x_train)
        y_output = tf.convert_to_tensor(y_output)
        loss = tf.keras.losses.categorical_crossentropy(y_train, y_output)
#        print(loss.shape)
#        grad = tf.gradients(loss, x_test)
        grad = tf.gradients(loss, x_train)
        grad_list = []
        for i in grad:
            print(i.shape)
            grad_list.append(tf.norm(i,2))
        grad_list=sess.run(grad_list)
        print(grad_list)
#    x_train = tf.convert_to_tensor(x_train)
        
#    with tf.Session() as sess:
#        grad_list=sess.run(grad[0])
        
        


#######################################################################
#                            save history                             #
#######################################################################
#    import pickle
#    with open(history_filename, 'wb') as f:
#        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    training(10)
    training(50)
    training(100)
    training(200)
    training(500)
    training(1000)
