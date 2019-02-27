import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import tensorflow_hub as hub
import numpy as np

ELMOSIZE = 1024

class ModelBasic:
	def __init__(self, args, textData, initializer=None, eager=False):
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
		self.eager = eager
		self.sample_weights = None
		self.weighted = None

		if self.args.elmo:
			self.embedding_size = ELMOSIZE
		else:
			self.embedding_size = self.args.embeddingSize


		self.v0 = None
		self.v1 = None
		self.v2 = None
		self.v3 = None
		self.v4 = None
		self.v5 = None
		self.v6 = None
		self.v7 = None

		if not eager:
			self.buildNetwork()

	def get_rnn_outputs(self, inputs, length, cell):
		"""

		:param inputs: [batch_size*n_turns, max_steps, embedding_size]
		:param length: [batch_size*n_turns]
		:param cell: LSTM cell
		:return: last_relevant_outputs: [batch_size*n_turn, hidden_size]
		"""
		outputs, state = tf.nn.dynamic_rnn(cell=cell,
		                                   inputs=inputs, sequence_length=length,
		                                   dtype=tf.float32)
		last_relevant_mask = tf.one_hot(indices=length - 1, depth=self.args.maxSteps, name='last_relevant',
		                                dtype=tf.int32)

		last_relevant_outputs = tf.boolean_mask(outputs, last_relevant_mask, name='last_relevant_outputs')

		return last_relevant_outputs

	def buildInputs(self):
		with tf.name_scope('placeholders'):
			# [batch_size, n_turn, max_steps]
			input_shape = [None, self.args.nTurn, self.args.maxSteps]

			if self.args.elmo:
				self.data = tf.placeholder(tf.string, shape=input_shape, name='data')
			else:
				self.data = tf.placeholder(tf.int32, shape=input_shape, name='data')
			# [batch_size, n_turn]
			self.length = tf.placeholder(tf.int32, shape=[None, self.args.nTurn], name='length')

			# [batch_size]
			self.labels = tf.placeholder(tf.int32, shape=[None,], name='labels')
			self.sample_weights = tf.placeholder(tf.float32, shape=[None,], name='sample_weights')

			# scalar
			self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
			self.dropOutRate = tf.placeholder(tf.float32, shape=(), name='dropOut')

			self.weighted = tf.placeholder(tf.bool, shape=(), name='weighted')

	def buildEmbeddings(self):
		with tf.name_scope('embedding_layer'):
			if not self.args.preEmbedding:
				print('Using randomly initialized embeddings!')
				embeddings = tf.get_variable(
					shape=[self.textData.getVocabularySize(), self.args.embeddingSize],
					initializer=tf.contrib.layers.xavier_initializer(),
					name='embeddings')
				# [batch_size, n_turn, max_steps, embedding_size]
				self.embedded = tf.nn.embedding_lookup(embeddings, self.data)
			elif not self.args.elmo:
				print('Using pretrained glove word embeddings!')
				embeddings = tf.Variable(self.textData.preTrainedEmbedding, name='embedding', dtype=tf.float32)
				# [batch_size, n_turn, max_steps, embedding_size]
				self.embedded = tf.nn.embedding_lookup(embeddings, self.data)
			else:
				# elmo not supported for eager execution

				elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.args.trainElmo)
				# [batch_size*n_turn, max_steps]
				data_elmo = tf.reshape(self.data, shape=[-1, self.args.maxSteps], name='data_elmo')
				# [batch_size*n_turn]
				length_elmo = tf.reshape(self.length, shape=[-1], name='length_elmo')
				# [batch_size*n_turn, elmo_size]
				self.embedded = elmo(
					inputs={
						"tokens": data_elmo,
						"sequence_len": length_elmo
					},
					signature="tokens",
					as_dict=True)['elmo']
				self.embedded = tf.reshape(self.embedded, shape=[self.batch_size, self.args.nTurn, self.args.maxSteps, ELMOSIZE], name='elmo_embedded')
			# [batch_size, n_turn, max_steps, embedding_size]
			self.embedded = tf.nn.dropout(self.embedded, self.dropOutRate, name='embedding_dropout')


	def buildNetwork(self):
		with tf.name_scope('inputs'):
			if not self.eager:
				self.buildInputs()
			self.buildEmbeddings()

		with tf.name_scope('rnn'):
			# [batch_size, hidden_size*3]
			outputs = self.buildRNN()

		with tf.name_scope('output'):
			weights = tf.get_variable(name='weights', shape=[self.args.hiddenSize*3, self.args.numClasses],
									  initializer=self.initializer)

			biases = tf.get_variable(name='biases', shape=[self.args.numClasses],
			                         initializer=self.initializer)
			# [batch_size, num_classes]
			logits = tf.nn.xw_plus_b(x=outputs, weights=weights, biases=biases)

		with tf.name_scope('predictions'):
			# [batch_size]
			self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
			# single number
			#labels = tf.slice(self.labels, begin=[0], size=[self.batch_size], name='labels')
			self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

		with tf.name_scope('loss'):
			# [batch_size]
			loss_equal = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='loss_equal')

			loss_weighted = tf.multiply(loss_equal, self.sample_weights, name='loss_weighted')

			weighted = tf.cast(self.weighted, tf.float32)
			loss = weighted*loss_weighted + (1.0-weighted)*loss_equal

			self.loss = tf.reduce_sum(loss)

		if self.eager:
			return self.loss, self.predictions, self.labels, self.corrects

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
				multiCell = []
				for i in range(self.args.rnnLayers):
					multiCell.append(get_cell(self.args.hiddenSize, self.dropOutRate))
				multiCell = tf.contrib.rnn.MultiRNNCell(multiCell, state_is_tuple=True)


			with tf.variable_scope('get_rnn_outputs', reuse=tf.AUTO_REUSE):
				# embedded: [batch_size, n_turn, max_steps, embedding_size]
				# length: [batch_size, n_turn]
				# outputs: [batch_size*n_turn, hidden_size]

				# embedded_reshaped: [batch_size*n_turn, max_steps, embedding_size]
				# length_reshaped: [batch_size*n_turn]
				embedded_reshaped = tf.reshape(self.embedded, shape=[self.batch_size*self.args.nTurn,
				                                                     self.args.maxSteps, self.embedding_size], name='embedded_reshaped')
				length_reshaped = tf.reshape(self.length, shape=[-1], name='length_reshaped')

				# outputs: [batch_size*n_turn, hidden_size]
				outputs = self.get_rnn_outputs(inputs=embedded_reshaped, length=length_reshaped, cell=multiCell)
				# outputs = tf.Print(outputs, data=[tf.shape(outputs)], message='outputs shape')
				outputs_concated = tf.reshape(outputs, shape=[-1, self.args.hiddenSize*self.args.nTurn], name='outputs_concated')

		return outputs_concated

	def step_eager(self, data, length, labels, test, sample_weights):
		"""
		not supported for training, currently only debugging
		:param data:
		:param length:
		:param labels:
		:param test:
		:param sample_weights:
		:return:
		"""

		self.labels = labels
		self.data = data
		self.length = length
		self.batch_size = len(labels)
		self.sample_weights = sample_weights
		self.weighted = self.args.weighted

		if not test:
			self.dropOutRate = self.args.dropOut
		else:
			self.dropOutRate = 1.0

	def step_graph(self, data, length, labels, test, sample_weights):
		"""
		:param data:
		:param length:
		:param labels:
		:param test:
		:param sample_weights:
		:return:
		"""
		feed_dict = dict()

		feed_dict[self.labels] = labels
		feed_dict[self.data] = np.asarray(data)
		feed_dict[self.length] = np.asarray(length)
		feed_dict[self.batch_size] = len(labels)
		feed_dict[self.sample_weights] = sample_weights
		feed_dict[self.weighted] = self.args.weighted

		if not test:
			feed_dict[self.dropOutRate] = self.args.dropOut
			ops = (self.optOp, self.loss, self.predictions, self.corrects)
		else:
			# during test, do not use drop out!!!!
			feed_dict[self.dropOutRate] = 1.0
			ops = (self.loss, self.predictions, self.corrects)

		return ops, feed_dict, labels, sample_weights

	def step(self, batch, test=False, eager=False):

		# [batch_size, n_turns, max_steps]
		data = []
		# [batch_size, n_turns]
		length = []
		# [batch_size]
		labels = []
		sample_weights = []

		for sample in batch.samples:
			labels.append(sample.label)
			if self.args.elmo:
				data.append(sample.sents)
			else:
				data.append(sample.data)
			length.append(sample.length)
			sample_weights.append(sample.sample_weight)

		data = np.asarray(data)
		length = np.asarray(length)
		labels = np.asarray(labels)
		sample_weights = np.asarray(sample_weights)
		sample_weights /= self.args.sampleWeight

		if eager:
			return self.step_eager(data=data, length=length, labels=labels, test=test, sample_weights=sample_weights)
		else:
			return self.step_graph(data=data, length=length, labels=labels, test=test, sample_weights=sample_weights)
