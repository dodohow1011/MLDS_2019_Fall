import tensorflow as tf
import numpy as np
import pickle
import sys
import os


WEIGHTS = {'layer_1':[],'layer_2':[],'layer_3':[],'layer_4':[]}


def load_mnist():
    (train_x, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_x.reshape((-1, 28*28))/255.0, train_y

def create_model(image_in):
    layer_1 = tf.layers.dense(image_in, units=16, activation=tf.nn.relu, name='layer_1')
    layer_2 = tf.layers.dense(layer_1, units=16, activation=tf.nn.relu, name='layer_2')
    layer_3 = tf.layers.dense(layer_2, units=16, activation=tf.nn.relu, name='layer_3')
    layer_4 = tf.layers.dense(layer_3, units=10, activation=tf.nn.softmax, name='layer_4')
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
    # if len(sys.argv) != 2:
      #  print ('usage: python3.5 mnist_dnn.py exp_?')
       # sys.exit()
    #Exp_filename = sys.argv[1]

    # load data
    train_x, train_y = load_mnist()

    # placeholders
    mnist_in  = tf.placeholder(tf.float32, [None, 28*28], name='flattened_image')
    mnist_out = tf.placeholder(tf.int64,   [None],        name='labels')

    # creat model
    with tf.variable_scope('model'): layers = create_model(mnist_in)

    # outpt of the computational graph
    classes  = tf.argmax(layers[-1], axis=1, name='output')
    equality = tf.equal(classes, mnist_out)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    sum_acc  = tf.summary.scalar('training_acc', accuracy)

    # loss, optimizer, summary writer
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(indices=mnist_out, depth=10),
                logits=layers[-1]
                )
            )
    sum_loss = tf.summary.scalar('training_loss', loss)
    
    gradients = tf.train.AdamOptimizer(learning_rate=1e-3).compute_gradients(loss)
    opt = tf.train.AdamOptimizer().apply_gradients(gradients)

#######################################################################
#                            model summary                            #
#######################################################################
    print ('Model Summary')
    for layer in layers:
        print (layer)

    batch_size = 64
    steps_per_epoch = train_x.shape[0] // batch_size
    gen = data_generator(train_x, train_y, batch_size)

    with tf.Session() as sess:

        # preparation
        grad_norms = []
        shit = [f for f in os.listdir('Tensorboard/')]
        for s in shit: os.remove(os.path.join('Tensorboard', s))
        writer = tf.summary.FileWriter('Tensorboard/', graph=sess.graph)
        all_vars = tf.global_variables()
        print (all_vars)
        sys.exit()
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
        for step in range(steps_per_epoch*22):
            curEpoch = 1+step//steps_per_epoch
            batch_x, batch_y = gen.__next__()
            feed_dict = {mnist_in: batch_x, mnist_out: batch_y}

            # retrieve weights every 3 epoch
            if step % (3*steps_per_epoch) == 0:
                target = [opt, loss, accuracy, sum_loss, sum_acc, gradients, layer_1_weights, layer_2_weights, layer_3_weights, layer_4_weights]
                _, curLoss, curAcc, s_l, s_a, grad, l1_w, l2_w, l3_w, l4_w = sess.run(target, feed_dict=feed_dict)
                record_weights(l1_w, l2_w, l3_w, l4_w)
            else:
                target = [opt, loss, accuracy, sum_loss, sum_acc, gradients]
                _, curLoss, curAcc, s_l, s_a, grad = sess.run(target, feed_dict=feed_dict)
            
            loss_list.append(curLoss)

            writer.add_summary(s_l, step)
            writer.add_summary(s_a, step)
            grad_norms.append(cal_norms(grad))
            pattern = 'Epoch {}, Step {}, Loss {:.6f}, Acc {:.6f}'
            print (pattern.format(curEpoch, step+1, curLoss, curAcc))
    
    # save the recorded weights
    # the first dimension is Epoch, and the second are the flattened weights
    # global WEIGHTS
   #  for layer, weights in WEIGHTS.items(): WEIGHTS[layer] = np.array(weights)
  #   with open(os.path.join(Exp_filename, 'Loss.pickle'), 'wb') as f:
 #        pickle.dump(loss_list, f, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(os.path.join(Exp_filename, 'Weights_dict.pickle'), 'wb') as f:
#         pickle.dump(WEIGHTS, f, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(os.path.join(Exp_filename, 'grad_norms.pickle'), 'wb') as f:
#         pickle.dump(grad_norms, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
