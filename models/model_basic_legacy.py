import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import numpy as np

class ModelBasic:
	def __init__(self, args, textData, initializer=None):
		print('Creating single lstm Model')
		self.args = args
		self.textData = textData

		self.dropOutRate = None
		self.initial_state = None
		self.learning_rate = None
		self.loss = None
		self.optOp = None
		self.labels = None
		self.input = None
		self.target = None
		self.length = None
		self.embedded = None
		self.predictions = None
		self.batch_size = None
		self.corrects = None
		self.initializer = initializer


		self.v0 = None
		self.v1 = None
		self.v2 = None
		self.v3 = None
		self.v4 = None
		self.v5 = None
		self.v6 = None
		self.v7 = None

		self.buildNetwork()

	def get_rnn_outputs(self, inputs, length, cell):
		# outputs: [batch_size, max_steps, hidden_size]
		outputs, state = tf.nn.dynamic_rnn(cell=cell,
		                                   inputs=inputs, sequence_length=length,
		                                   dtype=tf.float32)
		# [batch_size, max_steps]
		last_relevant_mask = tf.one_hot(indices=length - 1, depth=self.args.maxSteps, name='last_relevant',
		                                dtype=tf.int32)

		# [batch_size, hidden_size]
		last_relevant_outputs = tf.boolean_mask(outputs, last_relevant_mask, name='last_relevant_outputs')

		return last_relevant_outputs

	def buildInputs(self):
		with tf.name_scope('placeholders'):
			# [batchSize, maxSteps]
			input_shape = [None, self.args.maxSteps]

			self.data_0 = tf.placeholder(tf.int32, shape=input_shape, name='data_0')
			self.data_1 = tf.placeholder(tf.int32, shape=input_shape, name='data_1')
			self.data_2 = tf.placeholder(tf.int32, shape=input_shape, name='data_2')
			# [batch_size]
			self.length_0 = tf.placeholder(tf.int32, shape=[None,], name='length_0')
			self.length_1 = tf.placeholder(tf.int32, shape=[None,], name='length_1')
			self.length_2 = tf.placeholder(tf.int32, shape=[None,], name='length_2')


			# [batch_size]
			self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')

			# scalar
			self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

			self.dropOutRate = tf.placeholder(tf.float32, (), name='dropOut')

		with tf.name_scope('embedding_layer'):
			if not self.args.preEmbedding:
				print('Using randomly initialized embeddings!')
				embeddings = tf.get_variable(
					shape=[self.textData.getVocabularySize(), self.args.embeddingSize],
					initializer=tf.contrib.layers.xavier_initializer(),
					name='embeddings')
			else:
				print('Using pretrained word embeddings!')
				embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding', dtype=tf.float32)

			# [batchSize, maxSteps, embeddingSize]
			self.embedded_0 = tf.nn.embedding_lookup(embeddings, self.data_0)
			self.embedded_0 = tf.nn.dropout(self.embedded_0, self.dropOutRate, name='embedding_dropout_0')

			self.embedded_1 = tf.nn.embedding_lookup(embeddings, self.data_1)
			self.embedded_1 = tf.nn.dropout(self.embedded_1, self.dropOutRate, name='embedding_dropout_1')

			self.embedded_2 = tf.nn.embedding_lookup(embeddings, self.data_2)
			self.embedded_2 = tf.nn.dropout(self.embedded_2, self.dropOutRate, name='embedding_dropout_2')

	def buildNetwork(self):
		with tf.name_scope('inputs'):
			self.buildInputs()

		with tf.name_scope('rnn'):
			# [batch_size, hidden_size*3]
			outputs = self.buildRNN()

		with tf.name_scope('output'):
			weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize*3, self.args.numClasses],
									  initializer=self.initializer)

			biases = tf.get_variable(name='biases', shape=[self.args.numClasses],
			                         initializer=self.initializer)
			# [batchSize, numClasses]
			logits = tf.nn.xw_plus_b(x=outputs, weights=weights, biases=biases)
		with tf.name_scope('predictions'):
			# [batchSize]
			self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
			# single number
			#labels = tf.slice(self.labels, begin=[0], size=[self.batch_size], name='labels')
			self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

		with tf.name_scope('loss'):
			loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss')

			self.loss = tf.reduce_sum(loss)

		with tf.name_scope('backpropagation'):
			opt = tf.train.AdamOptimizer(learning_rate=self.args.learningRate, beta1=0.9, beta2=0.999,
											   epsilon=1e-08)
			self.optOp = opt.minimize(self.loss)

	def buildRNN(self):

		with tf.name_scope('lstm'):
			with tf.variable_scope('cell', reuse=False):

				def get_cell(hiddenSize, dropOutRate):
					print('building ordinary cell!')
					cell = BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
					cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
															 output_keep_prob=dropOutRate)
					return cell

				# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn

				cell = get_cell(self.args.hiddenSize, self.dropOutRate)
				# multiCell = []
				# for i in range(self.args.rnnLayers):
				# 	multiCell.append(get_cell(self.args.hiddenSize, self.dropOutRate))
				# multiCell = tf.contrib.rnn.MultiRNNCell(multiCell, state_is_tuple=True)

			with tf.variable_scope('get_rnn_outputs', reuse=tf.AUTO_REUSE):
				# [batch_size, hidden_size]
				output_0 = self.get_rnn_outputs(inputs=self.embedded_0, length=self.length_0, cell=cell)
				output_1 = self.get_rnn_outputs(inputs=self.embedded_1, length=self.length_1, cell=cell)
				output_2 = self.get_rnn_outputs(inputs=self.embedded_2, length=self.length_2, cell=cell)

		# [hidden_size*3]
		outputs_concated = tf.concat(values=[output_0, output_1, output_2], axis=1, name='outputs_concated')

		return outputs_concated

	def step(self, batch, test=False):
		feed_dict = {}

		# [batchSize, maxSteps]
		data_0 = []
		data_1 = []
		data_2 = []

		length_0 = []
		length_1 = []
		length_2 = []


		labels = []

		for sample in batch.samples:
			labels.append(sample.label)
			data_0.append(sample.data[0])
			data_1.append(sample.data[1])
			data_2.append(sample.data[2])

			length_0.append(sample.length[0])
			length_1.append(sample.length[1])
			length_2.append(sample.length[2])

		feed_dict[self.labels] = labels
		feed_dict[self.data_0] = np.asarray(data_0)
		feed_dict[self.data_1] = np.asarray(data_1)
		feed_dict[self.data_2] = np.asarray(data_2)

		feed_dict[self.length_0] = np.asarray(length_0)
		feed_dict[self.length_1] = np.asarray(length_1)
		feed_dict[self.length_2] = np.asarray(length_2)


		feed_dict[self.batch_size] = len(labels)

		if not test:
			feed_dict[self.dropOutRate] = self.args.dropOut
			ops = (self.optOp, self.loss, self.predictions, self.corrects)
		else:
			# during test, do not use drop out!!!!
			feed_dict[self.dropOutRate] = 1.0
			ops = (self.loss, self.predictions, self.corrects)

		return ops, feed_dict, labels
