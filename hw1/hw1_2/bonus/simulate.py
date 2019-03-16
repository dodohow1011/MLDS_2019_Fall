import sys
import numpy as np
import tensorflow as tf

class model():
	"""docstring for model"""
	def __init__(self, sess, epochs=100, learning_rate=5e-3, samples_per_point=100):
		super(model, self).__init__()
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.samples_per_point = samples_per_point
		self.sess = sess
		self.weights = []

		##########################
		## sin(3*pi*x)/(3*pi*x) ##
		##########################
		self.data_x = np.arange(0, 1, 0.005, dtype='f').reshape((-1, 1))
		self.data_y = np.sinc(3*self.data_x)

	def _create(self, train_x):
		x = tf.layers.dense(train_x, units=3, activation=tf.nn.relu, name='layer_1')
		x = tf.layers.dense(x,       units=3, activation=tf.nn.relu, name='layer_2')
		x = tf.layers.dense(x,       units=3, activation=tf.nn.relu, name='layer_3')
		x = tf.layers.dense(x,       units=1, activation=None,       name='layer_4')
		return x

	def _record_weights(self, l1_w, l2_w ,l3_w, l4_w, curLoss):
		self.weights.append(([l1_w, l2_w, l3_w, l4_w], curLoss))

	def _add_rand_noise(self, w, mean=0.0, stddev=1):
		shape = tf.shape(w)
		noise = tf.random_normal(shape, mean=mean, stddev=stddev, dtype=tf.float32)
		return tf.assign_add(w, noise)		

	def train(self):
		train_x = tf.placeholder(tf.float32, [None, 1], name='x')
		train_y = tf.placeholder(tf.float32, [None, 1], name='y')

		self.feed_dict = {train_x: self.data_x, train_y: self.data_y}

		with tf.variable_scope('model'): output_layer = self._create(train_x)

		self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_y - output_layer), reduction_indices=[1]))
		self.opt  = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver(max_to_keep=100)

		all_vars = tf.global_variables()
		def get_var(name):
			for i in range(len(all_vars)):
				if all_vars[i].name.startswith(name):
					return all_vars[i]
			return None

		for epoch in range(self.epochs):
			self.layer_1_weights = get_var('model/layer_1/kernel')
			self.layer_2_weights = get_var('model/layer_2/kernel')
			self.layer_3_weights = get_var('model/layer_3/kernel')
			self.layer_4_weights = get_var('model/layer_4/kernel')
			if epoch < 10:
				_, curLoss = self.sess.run([self.opt, self.loss], feed_dict=self.feed_dict)
			else:
				_, curLoss, l1_w, l2_w, l3_w, l4_w = self.sess.run([self.opt, self.loss, self.layer_1_weights, self.layer_2_weights, self.layer_3_weights, self.layer_4_weights], feed_dict=self.feed_dict)
				self.saver.save(self.sess, 'model/epoch_{}.ckpt'.format(epoch+1))
			print ('epoch {:3d}, loss {:.6f}'.format(epoch+1, curLoss))

	def sample_weights(self):
		# do this by loading back the checkpoints
		for weights, loss in self.weights:
			print (weights, loss)
			sampled_weights = []
			for w in weights:
				sampled_weights.append(self.sess.run(self._add_rand_noise(w)))
			print (sampled_weights, self.sess.run(self.loss, feed_dict=feed_dict))
		sys.exit()


def main():
	Model = model(tf.Session())
	Model.train()

if __name__ == '__main__':
	main()