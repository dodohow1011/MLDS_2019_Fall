import tensorflow as tf
import numpy as np
import pickle
import sys
import os
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.33
config.gpu_options.allow_growth = True

class Model():
    def __init__(self, batch_size, sess, model_dir):
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
            print ('loss {:.6f}, acc {:.6f}'.format(l, a))
            l_his.append(l)
            a_his.append(a)
        self.test_loss_ave = np.array(l_his).mean()
        self.test_acc_ave  = np.array(a_his).mean()
        print ('Ave Loss {}, Ave Acc {}'.format(self.test_loss_ave, self.test_acc_ave))
        print ('total parameters', self.total_param)

def main():
    if len(sys.argv) != 3:
        print ("usage: python3 random_mnist.py <gpu ID> <model directory>")
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    model = Model(batch_size=32, sess=tf.Session(config=config), model_dir=sys.argv[2])
    model.load_mnist()
    model.inference()

if __name__ == '__main__':
    main()
