import numpy as np
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
import sys
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth=True


def read():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
	# number = 10000
    # np.random.shuffle(y_train[:30000])

	# x_train = x_train[0:number]
	# y_train = y_train[0:number]

    x_train = x_train.reshape((-1,32,32,3))/255
    x_test = x_test.reshape((-1,32,32,3))/255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # y_train = np_utils.to_categorical(y_train, 10)
    # y_test = np_utils.to_categorical(y_test, 10)

    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)

def create_model(image_in):
    layer_1 = tf.layers.conv2d(image_in, filters=256, kernel_size=(3,3), activation=tf.nn.relu, name='layer_1')
    layer_2 = tf.layers.conv2d(layer_1, filters=256, kernel_size=(3,3), activation=tf.nn.relu, name='layer_2')
    max_pool_1 = tf.layers.max_pooling2d(layer_2, pool_size=(2,2), strides=2, name='max_pool_1')
    layer_3 = tf.layers.conv2d(max_pool_1, filters=256, kernel_size=(3,3), activation=tf.nn.relu, name='layer_3')
    layer_4 = tf.layers.conv2d(layer_3, filters=256, kernel_size=(3,3), activation=tf.nn.relu, name='layer_4')
    max_pool_2 = tf.layers.max_pooling2d(layer_4, pool_size=(2,2), strides=2, name='max_pool_2')
    flatten = tf.layers.flatten(max_pool_2, name='flatten')
    layer_5 = tf.layers.dense(flatten, units=10, activation=tf.nn.softmax, name='layer_5')
    return [layer_1, layer_2, layer_3, layer_4, layer_5], [max_pool_1, max_pool_2], flatten


def shuffle_data(train_x, train_y):
    print ('Shuffling data...', end='')
    p = np.random.permutation(train_x.shape[0])
    print ('Done')
    return train_x[p], train_y[p]

def data_generator(train_x, train_y, batch_size):
    batch_cout = train_x.shape[0] // batch_size
    while True:
        train_x, train_y = shuffle_data(train_x, train_y)
        for batch in range(batch_cout):
            data_x = list()
            data_y = list()
            for i in range(batch_size):
                data_x.append(train_x[batch*batch_size+i])
                data_y.append(train_y[batch*batch_size+i])
            yield np.array(data_x), np.array(data_y)

def main():
    train_x, train_y, test_x, test_y = read()
    
    mnist_in = tf.placeholder(tf.float32, [None, 32, 32, 3])
    mnist_out = tf.placeholder(tf.int64, [None])

    with tf.variable_scope('model'):
        layers, max_pool, flatten = create_model(mnist_in)

    
    classes = tf.argmax(layers[-1], axis=1)
    equality = tf.equal(classes, mnist_out)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices=mnist_out, depth=10), logits=layers[-1]))
    gradients = tf.train.AdamOptimizer(learning_rate=1e-3).compute_gradients(loss)
    opt = tf.train.AdamOptimizer(1e-3).apply_gradients(gradients)
    
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    
    # batch_size
    batch_size = 100
    steps_per_epoch = train_x.shape[0] // batch_size

    gen_train = data_generator(train_x, train_y, batch_size)
    
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        for step in range(steps_per_epoch*20):
            curEpoch = step//steps_per_epoch+1
            x, y = gen_train.__next__()
            target = [opt, gradients, loss, accuracy]

            _, _, Loss, Acc = sess.run(target, feed_dict={mnist_in: x, mnist_out: y})
            # if (step+1) % steps_per_epoch == 0:
            pattern = 'Epoch {}, Loss {:.6f}, Train_Acc {:.6f}, Val_acc '
            print (pattern.format(curEpoch, Loss, Acc), end='')
            
            train_loss_list.append(Loss)
        
            
            print ('{:.6f}'.format(sess.run(accuracy, feed_dict={mnist_in: test_x[:batch_size], mnist_out: test_y[:batch_size]})))


    with open('train_loss.pickle', 'wb') as f:
        pickle.dump(train_loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('test_loss.pickle', 'wb') as f1:
       #  pickle.dump(test_loss_list, f1, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('test_acc.pickle', 'wb') as f2:
       #  pickle.dump(test_acc_list, f2, protocol=pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    main()
