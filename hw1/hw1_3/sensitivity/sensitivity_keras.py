from keras import optimizers
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
import tensorflow.keras.backend as K
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))



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
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
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
        history = model.fit(x_train,y_train,batch_size=number,epochs=10)
        grad = tf.gradients(model.outputs,model.inputs)
        grad_norm = tf.norm(grad[0],ord='euclidean')
        grad_norm = sess.run(grad_norm,feed_dict={model.input: x_test})
        return grad_norm

        
        
if __name__ == '__main__':
    grad_list = []
    grad_list.append(training(10))
    grad_list.append(training(50))
    grad_list.append(training(100))
    grad_list.append(training(200))
    grad_list.append(training(500))
    grad_list.append(training(1000))
    grad_list.append(training(2000))
    grad_list.append(training(10000))
    print(grad_list)
    np.save("norm_model",grad_list)
