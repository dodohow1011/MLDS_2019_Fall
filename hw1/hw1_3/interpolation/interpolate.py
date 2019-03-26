import tensorflow as tf
import numpy as np
import pickle
import sys
import os
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
config.gpu_options.allow_growth = True

class Model():
    def __init__(self, batch_size, sess, out):
        self.batch_size = batch_size
        self.sess = sess
        self.out = out
        self.model_1 = {}
        self.model_2 = {}
        self.result = [] # stored as (alpha, train_loss, train_acc, test_loss, test_acc)

    def create_interpolation_ratio(self, _min, _max, _interval):
        self.interpolation_ratio = np.arange(_min, _max, _interval)

    def load_mnist(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        self.train_x = train_x.reshape((-1, 28, 28, 1))/255.0
        self.train_y = train_y
        self.test_x  = test_x.reshape((-1, 28, 28, 1))/255.0
        self.test_y  = test_y

    def create_model(self, image_in):
        conv_1 = tf.layers.conv2d(image_in, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, padding='same', name='conv_1')
        pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2, name='pool_1')
        conv_2 = tf.layers.conv2d(pool_1,   filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_2')
        pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2, name='pool_2')
        conv_3 = tf.layers.conv2d(pool_2,   filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_3')
        pool_3 = tf.layers.max_pooling2d(conv_3, 2, 2, name='pool_3')
        flat   = tf.layers.flatten(pool_3)
        dense_1 = tf.layers.dense(flat,    units=64, activation=tf.nn.relu, name='dense_1')
        dense_2 = tf.layers.dense(dense_1, units=64, activation=tf.nn.relu, name='dense_2')
        dense_3 = tf.layers.dense(dense_2, units=10, activation=tf.nn.relu, name='dense_3')
        return dense_3

    def data_generator(self, train_x, train_y, batch_size):
        batch_cout = train_x.shape[0] // batch_size
        while True:
            for batch in range(batch_cout):
                data_x = list()
                data_y = list()
                for i in range(batch_size):
                    data_x.append(train_x[batch*batch_size+i])
                    data_y.append(train_y[batch*batch_size+i])
                yield np.array(data_x), np.array(data_y)

    def create_graph(self):
        # placeholders
        self.mnist_in  = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
        self.mnist_out = tf.placeholder(tf.int64,   shape=[None],            name='labels')

        # creat model
        with tf.variable_scope('model'): probs = self.create_model(self.mnist_in)

        # outpt of the computational graph
        classes  = tf.argmax(probs, axis=1, name='output')
        equality = tf.equal(classes, self.mnist_out)
        self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

        # loss, optimizer
        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(indices=self.mnist_out, depth=10),
                    logits=probs
                    )
                )

    def load_model(self, model_1, model_2):
        self.create_graph()
        g = tf.get_default_graph()
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(model_1))
        print (model_1, 'successfully restored.')
        print ('extracting weights and bias...')
        self.model_1['model/conv_1/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/conv_1/kernel:0'))
        self.model_1['model/conv_1/bias:0']   = self.sess.run(g.get_tensor_by_name('model/conv_1/bias:0'))
        self.model_1['model/conv_2/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/conv_2/kernel:0'))
        self.model_1['model/conv_2/bias:0']   = self.sess.run(g.get_tensor_by_name('model/conv_2/bias:0'))
        self.model_1['model/conv_3/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/conv_3/kernel:0'))
        self.model_1['model/conv_3/bias:0']   = self.sess.run(g.get_tensor_by_name('model/conv_3/bias:0'))
        self.model_1['model/dense_1/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/dense_1/kernel:0'))
        self.model_1['model/dense_1/bias:0']   = self.sess.run(g.get_tensor_by_name('model/dense_1/bias:0'))
        self.model_1['model/dense_2/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/dense_2/kernel:0'))
        self.model_1['model/dense_2/bias:0']   = self.sess.run(g.get_tensor_by_name('model/dense_2/bias:0'))
        self.model_1['model/dense_3/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/dense_3/kernel:0'))
        self.model_1['model/dense_3/bias:0']   = self.sess.run(g.get_tensor_by_name('model/dense_3/bias:0'))

        saver.restore(self.sess, tf.train.latest_checkpoint(model_2))
        print (model_2, 'successfully restored.')
        print ('extracting weights and bias...')
        self.model_2['model/conv_1/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/conv_1/kernel:0'))
        self.model_2['model/conv_1/bias:0']   = self.sess.run(g.get_tensor_by_name('model/conv_1/bias:0'))
        self.model_2['model/conv_2/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/conv_2/kernel:0'))
        self.model_2['model/conv_2/bias:0']   = self.sess.run(g.get_tensor_by_name('model/conv_2/bias:0'))
        self.model_2['model/conv_3/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/conv_3/kernel:0'))
        self.model_2['model/conv_3/bias:0']   = self.sess.run(g.get_tensor_by_name('model/conv_3/bias:0'))
        self.model_2['model/dense_1/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/dense_1/kernel:0'))
        self.model_2['model/dense_1/bias:0']   = self.sess.run(g.get_tensor_by_name('model/dense_1/bias:0'))
        self.model_2['model/dense_2/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/dense_2/kernel:0'))
        self.model_2['model/dense_2/bias:0']   = self.sess.run(g.get_tensor_by_name('model/dense_2/bias:0'))
        self.model_2['model/dense_3/kernel:0'] = self.sess.run(g.get_tensor_by_name('model/dense_3/kernel:0'))
        self.model_2['model/dense_3/bias:0']   = self.sess.run(g.get_tensor_by_name('model/dense_3/bias:0'))

        self.keys = []
        for key, value in self.model_1.items():
            self.keys.append(key)
    
    def run_entire_interpolation(self):
        for alpha in self.interpolation_ratio:
            self.interpolate(alpha)

    def interpolate(self, alpha):
        print ('[interpolating model with alpha = {:.3f}]'.format(alpha))
        g = tf.get_default_graph()

        for key in self.keys:
            t = g.get_tensor_by_name(key)
            self.sess.run(tf.assign(t, tf.add(tf.multiply(self.model_1[key], 1-alpha), tf.multiply(self.model_2[key], alpha))))

        gen = self.data_generator(self.train_x, self.train_y, self.batch_size)
        train_loss = []
        train_acc  = []
        for i in range(self.train_x.shape[0] // self.batch_size):
            batch_x, batch_y = gen.__next__()
            l, a = self.sess.run([self.loss, self.accuracy], feed_dict={self.mnist_in: batch_x, self.mnist_out: batch_y})
            train_loss.append(l)
            train_acc.append(a)

        gen = self.data_generator(self.test_x, self.test_y, self.batch_size)
        test_loss = []
        test_acc  = []
        for i in range(self.test_x.shape[0] // self.batch_size):
            batch_x, batch_y = gen.__next__()
            l, a = self.sess.run([self.loss, self.accuracy], feed_dict={self.mnist_in: batch_x, self.mnist_out: batch_y})
            test_loss.append(l)
            test_acc.append(a)

        self.result.append((alpha, sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc), sum(test_loss)/len(test_loss), sum(test_acc)/len(test_acc)))

        with open(self.out, 'wb') as f:
            pickle.dump(self.result, f,  protocol=pickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) != 5:
        print ("usage: python3 interpolate.py <model_1 directory> <model_2 directory> <gpu ID> <output filename>")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[3])
    
    model_1 = sys.argv[1]
    model_2 = sys.argv[2]

    model = Model(batch_size=1000, sess=tf.Session(config=config), out=sys.argv[4])
    model.load_mnist()
    model.load_model(model_1, model_2)
    model.create_interpolation_ratio(-1, 2, 0.05)
    model.run_entire_interpolation()

if __name__ == '__main__':
    main()
