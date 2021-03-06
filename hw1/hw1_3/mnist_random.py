import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
import sys
import pickle
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


def read():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
	# number = 10000
    np.random.shuffle(y_train[:30000])
    # np.random.shuffle(y_test[:5000])
	# x_train = x_train[0:number]
	# y_train = y_train[0:number]

    x_train = x_train.reshape((-1,28,28,1))/255
    x_test = x_test.reshape((-1,28,28,1))/255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # y_train = np_utils.to_categorical(y_train, 10)
    # y_test = np_utils.to_categorical(y_test, 10)

    return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)

def create_model(image_in, is_training):
    layer_1 = tf.layers.conv2d(image_in, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='layer_1')
    # layer_1 = tf.layers.dropout(layer_1, 0.5)
    layer_1 = tf.layers.batch_normalization(layer_1, training=is_training)
    layer_2 = tf.layers.conv2d(layer_1, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='layer_2')
    max_pool_1 = tf.layers.max_pooling2d(layer_2, pool_size=[2,2], strides=2, name='max_pool_1')
    max_pool_1 = tf.layers.batch_normalization(max_pool_1, training=is_training)
    layer_3 = tf.layers.conv2d(max_pool_1, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='layer_3')
    layer_3 = tf.layers.batch_normalization(layer_3, training=is_training)
    # layer_3 = tf.layers.dropout(layer_3, 0.5)
    layer_4 = tf.layers.conv2d(layer_3, filters=256, kernel_size=[3,3], activation=tf.nn.relu, name='layer_4')
    # layer_4 = tf.layers.dropout(layer_4, 0.5)
    max_pool_2 = tf.layers.max_pooling2d(layer_4, pool_size=[2,2], strides=2, name='max_pool_2')

    max_pool_2 = tf.layers.batch_normalization(max_pool_2, training=is_training)
    flatten = tf.layers.flatten(max_pool_2, name='flatten')
    layer_5 = tf.layers.dense(flatten, units=10, activation=tf.nn.relu, name='layer_5')
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
    
    mnist_in = tf.placeholder(tf.float32, [None, 28, 28, 1])
    mnist_out = tf.placeholder(tf.int64, [None])
    is_training = tf.placeholder(tf.bool)

    with tf.variable_scope('model'):
        layers, max_pool, flatten = create_model(mnist_in, is_training)

    
    classes = tf.argmax(layers[-1], axis=1)
    equality = tf.equal(classes, mnist_out)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices=mnist_out, depth=10), logits=layers[-1]))
    gradients = tf.train.AdamOptimizer(learning_rate=1e-4).compute_gradients(loss)
    opt = tf.train.AdamOptimizer(1e-4).apply_gradients(gradients)
    
    loss_hist = []
    loss_list = []
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    # batch_size
    batch_size = 128
    steps_per_epoch = train_x.shape[0] // batch_size

    gen_train = data_generator(train_x, train_y, batch_size)
    gen_test = data_generator(test_x, test_y, batch_size)
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())
        for step in range(steps_per_epoch*500):
            curEpoch = step//steps_per_epoch+1
            x, y = gen_train.__next__()
            test_x, test_y = gen_test.__next__()
            target = [opt, gradients, loss, accuracy]

            _, _, Loss, Acc = sess.run(target, feed_dict={mnist_in: x, mnist_out: y, is_training: True})
            pattern = 'Epoch {}, Steps {}, Loss {:.6f}, Train_Acc {:.6f}, Test_Loss {:.6f}'
            testLoss, testAcc = sess.run([loss, accuracy], feed_dict={mnist_in: test_x, mnist_out: test_y, is_training: True})
            print (pattern.format(curEpoch, step+1, Loss, Acc, testAcc))            
            
            loss_hist.append(Loss)
            if (step+1) % steps_per_epoch == 0:
                
                train_loss.append(sum(loss_hist)/len(loss_hist))
                train_acc.append(Acc)
                test_loss.append(testLoss)
                test_acc.append(testAcc)
                
                # early stop
                if len(train_loss) >= 5:
                    loss_list = train_loss[-5:]
                    if np.argmin(np.array(loss_list)) == 0 and curEpoch > 100: break
                loss_hist = []
    
    # saving data
    with open('trainLoss.pickle', 'wb') as f:
        pickle.dump(train_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('trainAcc.pickle', 'wb') as f:
        pickle.dump(train_acc, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('testLoss.pickle', 'wb') as f:
        pickle.dump(test_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('testAcc.pickle', 'wb') as f:
        pickle.dump(test_acc, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
