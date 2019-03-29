import tensorflow as tf
import numpy as np
import pickle
import sys
import os
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
config.gpu_options.allow_growth = True

class Model():
    def __init__(self, batch_size, learning_rate, patience, filters, epoch, sess, out, tensorboard):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.filters = filters
        self.epoch = epoch
        self.sess = sess
        self.out = out
        self.tensorboard_path = tensorboard

    def create_interpolation_ratio(self, _min, _max, _interval):
        self.interpolation_ratio = np.arange(_min, _max, _interval)

    def load_mnist(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        self.train_x = train_x.reshape((-1, 28, 28, 1))/255.0
        self.train_y = train_y
        self.test_x  = test_x.reshape((-1, 28, 28, 1))/255.0
        self.test_y  = test_y
        self.random_label(0.5)

    def random_label(self, portion):
        np.random.shuffle(self.train_y[:int(self.train_y.shape[0]*portion)])

    def create_model(self, image_in, training):
        filters = self.filters
        conv_1 = tf.layers.conv2d(image_in, filters=filters, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_1')
        conv_1 = tf.layers.batch_normalization(conv_1, training=training)
        conv_1 = tf.layers.conv2d(conv_1,   filters=filters, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_1_1')
        pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2, name='pool_1')
        pool_1 = tf.layers.batch_normalization(pool_1, training=training)

        conv_2 = tf.layers.conv2d(pool_1,   filters=filters, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_2')
        conv_2 = tf.layers.batch_normalization(conv_2, training=training)
        conv_2 = tf.layers.conv2d(conv_2,   filters=filters, kernel_size=[3, 3], activation=tf.nn.relu, padding='same', name='conv_2_2')
        pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2, name='pool_2')
        pool_2 = tf.layers.batch_normalization(pool_2, training=training)

        flat   = tf.layers.flatten(pool_2)
        dense_1 = tf.layers.dense(flat, units=10, activation=tf.nn.relu, name='dense_1')
        self.total_param = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.all_variables()])
        return dense_1

    def shuffle(self, train_x, train_y):
        p = np.random.permutation(train_x.shape[0])
        return train_x[p], train_y[p]

    def data_generator(self, train_x, train_y, batch_size):
        batch_cout = train_x.shape[0] // batch_size
        while True:
            train_x, train_y = self.shuffle(train_x, train_y)
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
        self.training  = tf.placeholder(tf.bool)

        # creat model
        with tf.variable_scope('model'): probs = self.create_model(self.mnist_in, self.training)

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
        self.sum_loss = tf.summary.scalar('loss', self.loss)
        self.sum_acc  = tf.summary.scalar('acc',  self.accuracy)

        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.create_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        train_gen = self.data_generator(self.train_x, self.train_y, self.batch_size)
        test_gen  = self.data_generator(self.test_x, self.test_y, self.batch_size)
        writer    = tf.summary.FileWriter(self.tensorboard_path, graph=self.sess.graph)

        loss_10_epoch = []
        loss_history = []

        for step in range(self.train_x.shape[0] // self.batch_size * self.epoch):
            epo = step // (self.train_x.shape[0] // self.batch_size) + 1
            if step % (self.train_x.shape[0] // self.batch_size) == 0 and len(loss_history) > 0:
                loss_10_epoch.append(sum(loss_history)/len(loss_history))
                if len(loss_10_epoch) > self.patience:
                    loss_10_epoch = loss_10_epoch[-self.patience:]
                    if np.argmin(np.array(loss_10_epoch)) == 0 and epo > 100: break
                loss_history = []

            batch_x, batch_y = train_gen.__next__()
            _, curLoss, curAcc, s_l, s_a = self.sess.run([self.opt, self.loss, self.accuracy, self.sum_loss, self.sum_acc], feed_dict={self.mnist_in: batch_x, self.mnist_out: batch_y, self.training: True})
            writer.add_summary(s_l, step)
            writer.add_summary(s_a, step)
            print ('[Epoch {}] [Step {}] loss: {:.6f} acc: {:.6f} Parameters: {} loss_hist:'.format(epo, step, curLoss, curAcc, self.total_param), end='')
            for p in loss_10_epoch:
                print (' {:.5f}'.format(p), end='')
            print ('')
            loss_history.append(curLoss)
        print ('saving model...')
        self.saver.save(self.sess, os.path.join(self.out, "model.ckpt"), global_step=step)

def main():
    if len(sys.argv) != 4:
        print ("usage: python3 random_mnist.py <gpu ID> <patience> <filters>")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    patience    = int(sys.argv[2])
    filters     = int(sys.argv[3])

    model_dir   = 'model_{}'.format(filters)
    tensorboard = os.path.join(model_dir, 'tensorboard')
    if os.path.exists(model_dir): os.rmdir(model_dir)
    os.mkdir(model_dir)
    os.mkdir(tensorboard)

    out = model_dir

    model = Model(batch_size=128, learning_rate=5e-5, epoch=800, patience=patience, filters=filters, sess=tf.Session(config=config), out=out, tensorboard=tensorboard)
    model.load_mnist()
    model.train()

if __name__ == '__main__':
    main()
