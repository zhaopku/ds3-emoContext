import os
import tensorflow as tf

def shape(x, dim):
	return x.get_shape()[dim].value or tf.shape(x)[dim]

def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout, output_weights_initializer=None):
	if len(inputs.get_shape()) > 3:
		raise ValueError("FFNN with rank {} not supported".format(len(inputs.get_shape())))

	if len(inputs.get_shape()) == 3:
		batch_size = shape(inputs, 0)
		seqlen = shape(inputs, 1)
		emb_size = shape(inputs, 2)
		current_inputs = tf.reshape(inputs, [batch_size * seqlen, emb_size])
	else:
		current_inputs = inputs

	for i in range(num_hidden_layers):
		hidden_weights = tf.get_variable("hidden_weights_{}".format(i), [shape(current_inputs, 1), hidden_size])
		hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
		current_outputs = tf.nn.relu(tf.nn.xw_plus_b(current_inputs, hidden_weights, hidden_bias))

	if dropout is not None:
		current_outputs = tf.nn.dropout(current_outputs, dropout)
	current_inputs = current_outputs

	output_weights = tf.get_variable("output_weights", [shape(current_inputs, 1), output_size], initializer=output_weights_initializer)
	output_bias = tf.get_variable("output_bias", [output_size])
	outputs = tf.nn.xw_plus_b(current_inputs, output_weights, output_bias)

	if len(inputs.get_shape()) == 3:
		outputs = tf.reshape(outputs, [batch_size, seqlen, output_size])
	return outputs


def makeSummary(value_dict):
	return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k,v in value_dict.items()])


def constructFileName(args, prefix=None, tag=None, createDataSetName=False):

	if createDataSetName:
		file_name = ''
		file_name += prefix + '-'
		file_name += str(args.vocabSize) + '-'
		file_name += str(args.batchSize) + '-'
		# if args.augment:
		# 	file_name += 'aug-'
		file_name += str(args.maxSteps) + '.pkl'
		return file_name

	file_name = ''
	file_name += 'hSize_' + str(args.hiddenSize)
	file_name += '_maxSteps_' + str(args.maxSteps)
	file_name += '_d_' + str(args.dropOut)

	file_name += '_lr_' + str(args.learningRate)
	file_name += '_bt_' + str(args.batchSize)
	file_name += '_vS_' + str(args.vocabSize)
	file_name += '_pre_' + str(args.preEmbedding)
	file_name += '_elmo_' + str(args.elmo)
	file_name += '_trainElmo_' + str(args.trainElmo)
	file_name += '_sw_' + str(args.sampleWeight)
	file_name += '_w_' + str(args.weighted)
	file_name += '_nL_' + str(args.rnnLayers)
	file_name += '_nS_' + str(args.nLSTM)
	file_name += '_spEm_' + str(args.speakerEmbedSize)

	if args.model.find('hbmp') != -1:
		if args.model.find('share') != -1:
			file_name += '_hbmp_share'
		else:
			file_name += '_hbmp'
		if args.attn and args.selfattn:
			file_name += '_heads_' + str(args.heads)
			file_name += '_ctx_' + str(args.nContexts)
		file_name += '_independent_' + str(args.independent)
		file_name += '_univ_' + str(args.universal)

	if args.model.find('dcrnn') != -1:
		file_name += '_dcrnn_ffnn_' + str(args.ffnn)

	if args.model.find('transfer') != -1:
		file_name += '_trans_lambda_' + str(args.gamma)

	if tag != 'model':
		file_name += '_loadModel_' + str(args.loadModel)

	file_name = os.path.join(prefix, file_name)

	return file_name

def writeInfo(out, args):
	out.write('embeddingSize {}\n'.format(args.embeddingSize))
	out.write('hiddenSize {}\n'.format(args.hiddenSize))

	out.write('dataset {}\n'.format(args.dataset))

	out.write('maxSteps {}\n'.format(args.maxSteps))
	out.write('dropOut {}\n'.format(args.dropOut))

	out.write('learningRate {}\n'.format(args.learningRate))
	out.write('batchSize {}\n'.format(args.batchSize))
	out.write('epochs {}\n'.format(args.epochs))

	out.write('loadModel {}\n'.format(args.loadModel))

	out.write('vocabSize {}\n'.format(args.vocabSize))
	out.write('preEmbeddings {}\n'.format(args.preEmbedding))
	out.write('elmo {}\n'.format(args.elmo))
	out.write('trainElmo {}\n'.format(args.trainElmo))
	out.write('sample_weight {}\n'.format(args.sampleWeight))
	out.write('sample_weight {}\n'.format(args.weighted))
	out.write('rnnLayers {}\n'.format(args.rnnLayers))

	out.write('speakerEmbedSize {}\n'.format(args.speakerEmbedSize))
	out.write('nLSTM {}\n'.format(args.nLSTM))

	out.write('model {}\n'.format(args.model))
	out.write('universal {}\n'.format(args.universal))

	if args.model.find('dcrnn') != -1:
		out.write('dcrnn_ffnn {}\n'.format(args.ffnn))

	if args.model.find('hbmp') != -1:
		out.write('hbmp_attention {}\n'.format(args.attn))
		out.write('heads {}\n'.format(args.heads))

		out.write('self attention {}\n'.format(args.selfattn))
		out.write('nContexts {}\n'.format(args.nContexts))
		out.write('independent {}\n'.format(args.independent))

	if args.model.find('transfer') != -1:
		out.write('transfer_gamma {}\n'.format(args.gamma))