import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import tensorflow_hub as hub
import numpy as np

ELMOSIZE = 1024
UNIVERSAL_SIZE = 512

class ModelHBMPShare:
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

		if not eager:
			self.buildNetwork()

	@staticmethod
	def get_bilstm_outputs(inputs, length, cell_fw, cell_bw, initial_state_fw=None, initial_state_bw=None):
		"""

		:param inputs: [batch_size*n_turns, max_steps, embedding_size]
		:param length: [batch_size*n_turns]
		:param cell_fw: LSTM cell
		:param cell_bw
		:param initial_state_bw:
		:param initial_state_fw:
		:return: last_relevant_outputs: [batch_size*n_turn, hidden_size]
		"""

		# outputs: (outputs_fw, outputs_bw): [batch_size*n_turns, max_steps, hidden_size]
		# state: (state_fw, state_bw): [batch_size*n_turns, hidden_size]
		# state_fw: (cell_last_relevant, output_last_relevant)
		outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
		                                   inputs=inputs, sequence_length=length, initial_state_fw=initial_state_fw,
		                                                 initial_state_bw=initial_state_bw, dtype=tf.float32)

		return outputs, state

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

			# [batch_size, n_turn]
			self.sents = tf.placeholder(tf.string, shape=[None, self.args.nTurn], name='sents')

	def buildUniversalSentEmbeds(self):
		with tf.name_scope('universal_sent_embeds'):
			with tf.device('/gpu:0'):
				embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
				sents = tf.reshape(self.sents, shape=[self.batch_size*self.args.nTurn])
				embeddings = embed(sents)
				# [batch_size, n_turn, sent_embed_size]
			embeddings = tf.reshape(embeddings, shape=[self.batch_size, self.args.nTurn, UNIVERSAL_SIZE])

		return embeddings

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

	def max_pool(self, outputs, length):
		"""

		:param outputs: [batch_size*n_turn, max_steps, hidden_size]
		:param length: [batch_size*n_turn]
		:return: [batch_size*n_turn, hidden_size]
		"""

		mask = tf.sequence_mask(lengths=length, maxlen=self.args.maxSteps, dtype=tf.float32)
		mask = tf.expand_dims(mask, axis=-1)
		# [batch_size*n_turn, max_steps, hidden_size]
		mask = tf.tile(mask, multiples=[1, 1, self.args.hiddenSize], name='mask')

		outputs_masked = tf.add(outputs, tf.log(mask), name='outputs_masked')

		# [batch_size * n_turn, hidden_size]
		outputs_pooled = tf.reduce_max(outputs_masked, axis=1, name='outputs_pooled')

		return outputs_pooled

	def buildNetwork(self):
		with tf.name_scope('inputs'):
			if not self.eager:
				self.buildInputs()
			self.buildEmbeddings()

		with tf.variable_scope('hbmp', reuse=False):
			outputs = []
			state_fw = None
			state_bw = None
			for i in range(self.args.nLSTM):
				# outputs_fw_pooled, outputs_bw_pooled: [batch_size*n_turn, hidden_size]
				with tf.variable_scope('hbmp_'+str(i), reuse=False):
					# should be three different bi-lstm
					(outputs_fw_pooled, outputs_bw_pooled), (state_fw, state_bw) = self.buildRNN(initial_state_fw=state_fw,
				                                                                             initial_state_bw=state_bw)
				# [batch_size*n_turn, hidden_size*2]
				outputs_pooled = tf.concat(values=[outputs_fw_pooled, outputs_bw_pooled], axis=-1)
				outputs.append(outputs_pooled)

			# outputs: [nLSTM, batch_size*n_turn, hidden_size*2]
			outputs = tf.stack(outputs)
			# outputs: [batch_size*n_turn, nLSTM, hidden_size*2]
			outputs = tf.transpose(outputs, perm=[1, 0, 2], name='outputs')
			# outputs: [batch_size, n_turn, nLSTM*hidden_size*2]
			concated_outputs = tf.reshape(outputs, shape=[self.batch_size, self.args.nTurn, self.args.nLSTM*self.args.hiddenSize*2],
			                              name='concated_outputs')

		if self.args.universal:
			with tf.name_scope('universal_embeds'):
				# [batch_size, n_turn, universal_size]
				universal_embeds = self.buildUniversalSentEmbeds()

				concated_outputs = tf.concat([universal_embeds, concated_outputs], axis=-1, name='concated_outputs')

		with tf.name_scope('speaker'):
			# number of speakers fixed to 2
			# [2, speaker_embed_size]

			speaker_embedding = tf.get_variable(name='speaker_embedding', shape=[2, self.args.speakerEmbedSize])

			# [3, speaker_embed_size]
			speaker_embedded = tf.gather(speaker_embedding, indices=[0, 1, 0])

			# [batch_size, n_turn, speaker_embed_size]
			speaker_embedded = tf.tile(tf.expand_dims(speaker_embedded, 0), multiples=[self.batch_size, 1, 1], name='speaker_embeded')

		with tf.name_scope('concated_output'):


			# remove liwc features

			if self.args.speakerEmbedSize > 0:
				# [batch_size, n_turn, hidden_size*2*3 + speaker_embed_size + 93]
				concated_outputs = tf.concat(values=[concated_outputs, speaker_embedded], axis=-1,
				                            name='concated_output_speaker')

			concated_size = self.args.nTurn*(self.args.hiddenSize*self.args.nLSTM*2+self.args.speakerEmbedSize)
			if self.args.universal:
				concated_size += UNIVERSAL_SIZE*self.args.nTurn
			concated_sentences = tf.reshape(concated_outputs, shape=[self.batch_size,
			                                                         concated_size],
			                                                        name='concated_sentences')
			# add dropout for final vector
			concated_sentences = tf.nn.dropout(concated_sentences, keep_prob=self.dropOutRate)
		with tf.name_scope('length'):
			# TODO: consider adding length as a feature
			pass

		with tf.name_scope('output'):
			weights = tf.get_variable(name='weights', shape=[concated_size, self.args.emoClasses],
									  initializer=self.initializer)

			biases = tf.get_variable(name='biases', shape=[self.args.emoClasses],
			                         initializer=self.initializer)
			# [batch_size, num_classes]
			logits = tf.nn.xw_plus_b(x=concated_sentences, weights=weights, biases=biases)

		with tf.name_scope('predictions'):
			# [batch_size]
			self.predictions = tf.argmax(logits, axis=-1, name='predictions', output_type=tf.int32)
			# single number
			#labels = tf.slice(self.labels, begin=[0], size=[self.batch_size], name='labels')
			self.corrects = tf.reduce_sum(tf.cast(tf.equal(self.predictions, self.labels), tf.int32), name='corrects')

		with tf.name_scope('loss'):
			trainable_variables = tf.trainable_variables()

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

	def buildRNN(self, initial_state_fw=None, initial_state_bw=None):

		with tf.name_scope('lstm'):
			with tf.variable_scope('cell', reuse=False):

				def get_cell(hiddenSize, dropOutRate):
					#print('building ordinary cell!')
					cell = BasicLSTMCell(num_units=hiddenSize, state_is_tuple=True)
					cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropOutRate,
															 output_keep_prob=dropOutRate)
					return cell

				# https://stackoverflow.com/questions/47371608/cannot-stack-lstm-with-multirnncell-and-dynamic-rnn
				multiCell_fw = []
				for i in range(self.args.rnnLayers):
					multiCell_fw.append(get_cell(self.args.hiddenSize, self.dropOutRate))
				multiCell_fw = tf.contrib.rnn.MultiRNNCell(multiCell_fw, state_is_tuple=True)

				multiCell_bw = []
				for i in range(self.args.rnnLayers):
					multiCell_bw.append(get_cell(self.args.hiddenSize, self.dropOutRate))
				multiCell_bw = tf.contrib.rnn.MultiRNNCell(multiCell_bw, state_is_tuple=True)


			with tf.variable_scope('get_rnn_outputs', reuse=tf.AUTO_REUSE):
				# embedded: [batch_size, n_turn, max_steps, embedding_size]
				# length: [batch_size, n_turn]
				# outputs: [batch_size*n_turn, hidden_size]

				# embedded_reshaped: [batch_size*n_turn, max_steps, embedding_size]
				# length_reshaped: [batch_size*n_turn]
				embedded_reshaped = tf.reshape(self.embedded, shape=[self.batch_size*self.args.nTurn,
				                                                     self.args.maxSteps, self.embedding_size], name='embedded_reshaped')
				length_reshaped = tf.reshape(self.length, shape=[-1], name='length_reshaped')

				# outputs: (outputs_fw, outputs_bw), [batch_size*n_turn, max_steps, hidden_size]
				(outputs_fw, outputs_bw), (state_fw, state_bw) \
					= self.get_bilstm_outputs(inputs=embedded_reshaped, length=length_reshaped, cell_fw=multiCell_fw,
				                                         cell_bw=multiCell_bw, initial_state_fw=initial_state_fw,
				                                         initial_state_bw=initial_state_bw)


				outputs_fw_pooled = self.max_pool(outputs_fw, length=length_reshaped)

				outputs_bw_pooled = self.max_pool(outputs_bw, length=length_reshaped)


		return (outputs_fw_pooled, outputs_bw_pooled), (state_fw, state_bw)

	def step_eager(self, data, length, labels, test, sample_weights, sents):
		"""
		not supported for training, currently only debugging
		:param data:
		:param length:
		:param labels:
		:param test:
		:param sample_weights:
		:param sents
		:return:
		"""

		self.labels = labels
		self.data = data
		self.length = length
		self.batch_size = len(labels)
		self.sample_weights = sample_weights
		self.weighted = self.args.weighted
		self.sents = sents
		if not test:
			self.dropOutRate = self.args.dropOut
		else:
			self.dropOutRate = 1.0

	def step_graph(self, data, length, labels, test, sample_weights, sents):
		"""
		:param data:
		:param length:
		:param labels:
		:param test:
		:param sample_weights:
		:param sents
		:return:
		"""
		feed_dict = dict()

		feed_dict[self.labels] = labels
		feed_dict[self.data] = np.asarray(data)
		feed_dict[self.length] = np.asarray(length)
		feed_dict[self.batch_size] = len(labels)
		feed_dict[self.sample_weights] = sample_weights
		feed_dict[self.weighted] = self.args.weighted
		feed_dict[self.sents] = sents
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

		# [batch_size, n_turns]
		sents = []
		for sample in batch.samples:
			labels.append(sample.label)
			if self.args.elmo:
				data.append(sample.sents)
			else:
				data.append(sample.data)

			cur_sents = []
			for idx, sent in enumerate(sample.sents):
				cur_sents.append(' '.join(sent[:sample.length[idx]]).strip())
			assert len(cur_sents) == 3
			sents.append(cur_sents)
			length.append(sample.length)
			sample_weights.append(sample.sample_weight)

		data = np.asarray(data)
		length = np.asarray(length)
		labels = np.asarray(labels)
		sample_weights = np.asarray(sample_weights)
		sample_weights /= self.args.sampleWeight

		if eager:
			return self.step_eager(data=data, length=length, labels=labels, test=test, sample_weights=sample_weights, sents=sents)
		else:
			return self.step_graph(data=data, length=length, labels=labels, test=test, sample_weights=sample_weights, sents=sents)
