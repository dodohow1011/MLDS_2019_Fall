import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
config.gpu_options.allow_growth = True

class Model():
    def __init__(self, batch_size, sess, model_dir):
        print ("Model:", model_dir, end='')
        self.batch_size = batch_size
        self.sess = sess
        self.model_dir = model_dir

        f = [x for x in os.listdir(self.model_dir) if x.endswith('meta')]
        self.model_path = os.path.join(self.model_dir, f[0])

    def load_mnist(self):
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        self.train_x = train_x.reshape((-1, 28, 28, 1))/255.0
        self.train_y = train_y
        self.test_x  = test_x.reshape((-1, 28, 28, 1))/255.0
        self.test_y  = test_y

    def inference(self):
        self.saver = tf.train.import_meta_graph(self.model_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
        g = tf.get_default_graph()
        
        mnist_in  = g.get_tensor_by_name('images:0')
        mnist_out = g.get_tensor_by_name('labels:0')
        training  = g.get_tensor_by_name('Placeholder:0')
        acc       = g.get_tensor_by_name('Mean_1:0')
        loss      = g.get_tensor_by_name('Mean:0')

        self.total_param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        steps = self.test_x.shape[0] // self.batch_size
        l_his = []
        a_his = []

        for s in range(steps):
            l, a = self.sess.run([acc, loss], feed_dict={mnist_in: self.test_x[self.batch_size*s:self.batch_size*(s+1)], mnist_out: self.test_y[self.batch_size*s:self.batch_size*(s+1)], training: True})
            l_his.append(l)
            a_his.append(a)
        self.test_loss_ave = np.array(l_his).mean()
        self.test_acc_ave  = np.array(a_his).mean()
        print (', Total parameters', self.total_param, end='')
        print (', Ave Loss {}, Ave Acc {}'.format(self.test_loss_ave, self.test_acc_ave))
        tf.reset_default_graph()

def main():
    if len(sys.argv) != 2:
        print ("usage: python3 plot_result.py <gpu ID>")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    models = [f for f in os.listdir('./') if f.startswith('model_')]
    training_acc = []
    training_loss = []
    testing_acc = []
    testing_loss = []
    param_num = []

    for M in models:
        train_acc_file = os.path.join(M, 'run-.-tag-acc.csv')
        train_loss_file = os.path.join(M, 'run-.-tag-loss.csv')
        acc_data = []
        loss_data = []
        with open(train_acc_file, 'r') as f:
            skip = True
            for line in f:
                if skip:
                    skip = False
                    continue
                value = line.strip().split(',')[-1]
                acc_data.append(float(value))
        with open(train_loss_file, 'r') as f:
            skip = True
            for line in f:
                if skip:
                    skip = False
                    continue
                value = line.strip().split(',')[-1]
                loss_data.append(float(value))
        acc_data = np.array(acc_data[-400:]).mean()
        loss_data = np.array(loss_data[-400:]).mean()
        training_acc.append(acc_data)
        training_loss.append(loss_data)

        sess = tf.Session(config=config)
        model = Model(batch_size=1000, sess=sess, model_dir=M)
        model.load_mnist()
        model.inference()
        testing_acc.append(model.test_acc_ave)
        testing_loss.append(model.test_loss_ave)
        param_num.append(model.total_param)

    plt.figure(1)
    plt.plot(param_num, training_loss, 'b.')
    plt.plot(param_num, testing_loss, 'r.')
    plt.xlabel('number of parameters')
    plt.ylabel('loss')
    plt.title('Model Loss')
    plt.legend(['training loss', 'testing loss'])
    plt.savefig('Model_Loss.png')

    plt.figure(2)
    plt.plot(param_num, training_acc, 'b.')
    plt.plot(param_num, testing_acc, 'r.')
    plt.xlabel('number of parameters')
    plt.ylabel('accuracy')
    plt.title('Model Accuracy')
    plt.legend(['training acc', 'testing acc'])
    plt.savefig('Model_Acc.png')

if __name__ == '__main__':
    main()
