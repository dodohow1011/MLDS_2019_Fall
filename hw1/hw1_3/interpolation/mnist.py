import tensorflow as tf
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[3])
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
config.gpu_options.allow_growth = True


def load_mnist():
    (train_x, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_x.reshape((-1, 28, 28, 1))/255.0, train_y

def create_model(image_in):
    conv_1 = tf.layers.conv2d(image_in, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, padding='same', name='conv_1')
    print ('conv_1', conv_1.shape)
    pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2, name='pool_1')
    print ('pool_1', pool_1.shape)
    conv_2 = tf.layers.conv2d(pool_1,   filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_2')
    print ('conv_2', conv_2.shape)
    pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2, name='pool_2')
    print ('pool_2', pool_2.shape)
    conv_3 = tf.layers.conv2d(pool_2,   filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_3')
    print ('conv_3', conv_3.shape)
    pool_3 = tf.layers.max_pooling2d(conv_3, 2, 2, name='pool_3')
    print ('pool_3', pool_3.shape)
    flat   = tf.layers.flatten(pool_3)
    print ('flat', flat.shape)
    dense_1 = tf.layers.dense(flat,    units=64, activation=tf.nn.relu, name='dense_1')
    print ('dense_1', dense_1.shape)
    dense_2 = tf.layers.dense(dense_1, units=64, activation=tf.nn.relu, name='dense_2')
    print ('dense_2', dense_2.shape)
    dense_3 = tf.layers.dense(dense_2, units=10, activation=tf.nn.relu, name='dense_3')
    print ('dense_3', dense_3.shape)
    return dense_3

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
    if len(sys.argv) != 4:
        print ("usage: python3 mnist.py <batch_size> <learning rate> <gpu ID>")
        sys.exit()

    # load data
    train_x, train_y = load_mnist()

    # placeholders
    mnist_in  = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
    mnist_out = tf.placeholder(tf.int64,   shape=[None],            name='labels')

    # creat model
    with tf.variable_scope('model'): probs = create_model(mnist_in)

    # outpt of the computational graph
    classes  = tf.argmax(probs, axis=1, name='output')
    equality = tf.equal(classes, mnist_out)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    # loss, optimizer
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(indices=mnist_out, depth=10),
                logits=probs
                )
            )
    
    opt = tf.train.AdamOptimizer(learning_rate=float(sys.argv[2])).minimize(loss)

    batch_size = int(sys.argv[1])
    steps_per_epoch = train_x.shape[0] // batch_size
    gen = data_generator(train_x, train_y, batch_size)

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        for step in range(steps_per_epoch*20):
            curEpoch = 1+step//steps_per_epoch
            batch_x, batch_y = gen.__next__()
            feed_dict = {mnist_in: batch_x, mnist_out: batch_y}
            target = [opt, loss, accuracy]
            _, curLoss, curAcc = sess.run(target, feed_dict=feed_dict)
            pattern = 'Epoch {}, Step {}, Loss {:.6f}, Acc {:.6f}'
            print (pattern.format(curEpoch, step+1, curLoss, curAcc))

        model_path = 'batchsize_{}_lr_{}'.format(batch_size, float(sys.argv[2]))
        if not os.path.exists(model_path): os.mkdir(model_path)
        saver.save(sess, os.path.join(model_path, "model.ckpt"))

if __name__ == '__main__':
    main()
