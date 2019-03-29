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
config.gpu_options.per_process_gpu_memory_fraction = 1.0
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

def deep2():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.summary()
    return model


def training(number):
    x_train, y_train, x_test, y_test = read()
    model = deep2()
    with tf.Session() as sess:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(x_train,y_train,batch_size=2**number,epochs=10,validation_data=(x_test,y_test))
        y_train = tf.convert_to_tensor(y_train)
        y_output = tf.convert_to_tensor(model.outputs)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train,logits=y_output))
        grad = tf.gradients(ce,model.input)
        grad_norm = tf.norm(grad[0],ord='euclidean')
        grad_norm = sess.run(grad_norm,feed_dict={model.input: x_train})
        record = np.array([history.history["loss"][-1],history.history["acc"][-1],history.history["val_loss"][-1],history.history["val_acc"][-1]])
    return grad_norm, record
        
if __name__ == '__main__':
    grad_list = []
    loss_list = []
    for i in range(4,12,1):
        norm, _record = training(i)
        grad_list.append(norm)
        loss_list.append(_record)
    np.save("norm_model",grad_list)
    print(grad_list)
    np.save("loss_model",loss_list)
