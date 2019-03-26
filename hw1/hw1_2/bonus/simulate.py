import tensorflow as tf
import numpy as np
import pickle
import math
import sys
import os

PI = math.pi
EPOCH = 200

def create_model(x_in):
    layer_1 = tf.layers.dense(x_in,    units=15, activation=tf.nn.relu, name='layer_1')
    layer_2 = tf.layers.dense(layer_1, units=1, activation=None, name='layer_2')
    return [layer_1, layer_2]

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    train_x = np.arange(0,1,0.005).reshape((-1,1))
    train_y = np.array([math.sin(3*PI*x)/(3*PI*x) for x in train_x]).reshape((-1,1))
    train_y[0] = 1

    # placeholders
    x_in  = tf.placeholder(tf.float32, [None, 1], name='x')
    y_out = tf.placeholder(tf.float32, [None, 1], name='y')

    # creat model
    with tf.variable_scope('model'): layers = create_model(x_in)

    # outpt of the computational graph
    out = layers[-1]

    # loss, optimizer, summary writer
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_out - out), reduction_indices=[1]))
    sum_loss = tf.summary.scalar('training_loss', loss)
    
    '''
    gradients = tf.train.AdamOptimizer(learning_rate=1e-3).compute_gradients(loss)
    opt = tf.train.AdamOptimizer(learning_rate=1e-3).apply_gradients(gradients)
    '''
    opt = tf.train.GradientDescentOptimizer(learning_rate=5e-3).minimize(loss)

    with tf.Session() as sess:

        # preparation
        all_vars = tf.global_variables()
        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None
        sess.run(tf.global_variables_initializer())

        layer_1_weights = get_var('model/layer_1/kernel')
        layer_2_weights = get_var('model/layer_2/kernel')

        for epoch in range(EPOCH):
            feed_dict = {x_in: train_x, y_out: train_y}

            # retrieve weights every 3 epoch
            if 0:
                target = [opt, loss, layer_1_weights, layer_2_weights]
                _, curLoss, l1_w, l2_w = sess.run(target, feed_dict=feed_dict)
            else:
                target = [opt, loss]
                _, curLoss = sess.run(target, feed_dict=feed_dict)

            pattern = 'Epoch {}, Loss {:.6f}'
            print (pattern.format(epoch+1, curLoss))

if __name__ == '__main__':
    main()
