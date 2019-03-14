import tensorflow as tf
import numpy as np
import pickle
import math
import sys
import os

PI = math.pi
WEIGHTS = {'layer_1':[], 'layer_2':[], 'layer_3':[], 'layer_4':[]}


def create_model(x_in):
    layer_1 = tf.layers.dense(x_in,    units=100, activation=tf.nn.relu, name='layer_1')
    layer_2 = tf.layers.dense(layer_1, units=100, activation=tf.nn.relu, name='layer_2')
    layer_3 = tf.layers.dense(layer_2, units=100, activation=tf.nn.relu, name='layer_3')
    layer_4 = tf.layers.dense(layer_3, units=1, activation=None, name='layer_4')
    return [layer_1, layer_2, layer_3, layer_4]

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

def record_weights(l1_w, l2_w, l3_w, l4_w):
    print ('recording weights')
    global WEIGHTS
    WEIGHTS['layer_1'].append(l1_w)
    WEIGHTS['layer_2'].append(l2_w)
    WEIGHTS['layer_3'].append(l3_w)
    WEIGHTS['layer_4'].append(l4_w)

def cal_norms(gradients):
    grad_all = 0
    for grad, var in gradients:
        if grad is not None:
            grad_all += (grad.reshape(-1) ** 2).sum()
    return grad_all ** 0.5

#def PCA():
   # numpy.vstack((WEIGHTS['layer_1'],WEIGHTS['layer_2']))

def main():
    if len(sys.argv) != 2:
       print ('usage: python3.5 func_dnn.py func_exp_?')
       sys.exit()
    Exp_filename = sys.argv[1]

    # load data
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
    
    gradients = tf.train.AdamOptimizer(learning_rate=1e-4).compute_gradients(loss)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4).apply_gradients(gradients)

    batch_size = 1
    steps_per_epoch = train_x.shape[0] // batch_size
    gen = data_generator(train_x, train_y, batch_size)

    with tf.Session() as sess:

        # preparation
        grad_norms = []
        all_vars = tf.global_variables()
        def get_var(name):
            for i in range(len(all_vars)):
                if all_vars[i].name.startswith(name):
                    return all_vars[i]
            return None
        sess.run(tf.global_variables_initializer())

        layer_1_weights = get_var('model/layer_1/kernel')
        layer_2_weights = get_var('model/layer_2/kernel')
        layer_3_weights = get_var('model/layer_3/kernel')
        layer_4_weights = get_var('model/layer_4/kernel')
        
        loss_list = []
        for step in range(steps_per_epoch*60):
            curEpoch = 1+step//steps_per_epoch
            batch_x, batch_y = gen.__next__()
            feed_dict = {x_in: batch_x, y_out: batch_y}

            # retrieve weights every 3 epoch
            if step % (3*steps_per_epoch) == 0:
                target = [opt, loss, gradients, layer_1_weights, layer_2_weights, layer_3_weights, layer_4_weights]
                _, curLoss, grad, l1_w, l2_w, l3_w, l4_w = sess.run(target, feed_dict=feed_dict)
                record_weights(l1_w, l2_w, l3_w, l4_w)
            else:
                target = [opt, loss, gradients]
                _, curLoss, grad = sess.run(target, feed_dict=feed_dict)
            
            loss_list.append(curLoss)

            grad_norms.append(cal_norms(grad))
            pattern = 'Epoch {}, Step {}, Loss {:.6f}'
            print (pattern.format(curEpoch, step+1, curLoss))
    
    # save the recorded weights
    # the first dimension is Epoch, and the second are the flattened weights
    global WEIGHTS
    for layer, weights in WEIGHTS.items(): WEIGHTS[layer] = np.array(weights)
    with open(os.path.join(Exp_filename, 'Loss.pickle'), 'wb') as f:
        pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(Exp_filename, 'Weights_dict.pickle'), 'wb') as f:
        pickle.dump(WEIGHTS, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(Exp_filename, 'grad_norms.pickle'), 'wb') as f:
        pickle.dump(grad_norms, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
