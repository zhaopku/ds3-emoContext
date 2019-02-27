import tensorflow as tf
import argparse
from models import utils
import os
from models.textData import TextData
from tqdm import tqdm
from models.model_basic import ModelBasic
import pickle as p
from sklearn.metrics import f1_score, precision_recall_fscore_support
from models.model_hbmp import ModelHBMP
from models.model_hbmp_share import ModelHBMPShare
import numpy as np
import csv

from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

class Train:
	def __init__(self):
		self.args = None

		self.textData = None
		self.model = None
		self.outFile = None
		self.sess = None
		self.saver = None
		self.model_name = None
		self.model_path = None
		self.globalStep = 0
		self.summaryDir = None
		self.testOutFile = None
		self.summaryWriter = None
		self.mergedSummary = None


	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		parser.add_argument('--resultDir', type=str, default='result', help='result directory')
		parser.add_argument('--testDir', type=str, default='test_result')
		# data location
		dataArgs = parser.add_argument_group('Dataset options')

		dataArgs.add_argument('--summaryDir', type=str, default='summaries')
		dataArgs.add_argument('--datasetName', type=str, default='dataset', help='a TextData object')

		dataArgs.add_argument('--dataDir', type=str, default='data', help='dataset directory, save pkl here')
		dataArgs.add_argument('--dataset', type=str, default='emo')
		dataArgs.add_argument('--trainFile', type=str, default='train.txt')
		dataArgs.add_argument('--valFile', type=str, default='dev.txt')
		dataArgs.add_argument('--testFile', type=str, default='test.txt')

		dataArgs.add_argument('--trainLiwcFile', type=str, default='train_liwc.csv')
		dataArgs.add_argument('--valLiwcFile', type=str, default='dev_liwc.csv')
		dataArgs.add_argument('--testLiwcFile', type=str, default='dev_liwc.csv')

		dataArgs.add_argument('--embeddingFile', type=str, default='glove.840B.300d.txt')
		dataArgs.add_argument('--vocabSize', type=int, default=-1, help='vocab size, use the most frequent words')

		dataArgs.add_argument('--snliDir', type=str, default='snli')
		dataArgs.add_argument('--trainSnliFile', type=str, default='train_snli.txt')
		dataArgs.add_argument('--valSnliFile', type=str, default='dev_snli.txt')
		dataArgs.add_argument('--testSnliFile', type=str, default='test_snli.txt')

		# neural network options
		nnArgs = parser.add_argument_group('Network options')
		nnArgs.add_argument('--embeddingSize', type=int, default=300)
		nnArgs.add_argument('--hiddenSize', type=int, default=300, help='hiddenSize for RNN sentence encoder')
		nnArgs.add_argument('--rnnLayers', type=int, default=1, help='number of RNN layers, fix to 1 in the DCRNN model')
		nnArgs.add_argument('--maxSteps', type=int, default=30)
		nnArgs.add_argument('--emoClasses', type=int, default=4)
		nnArgs.add_argument('--snliClasses', type=int, default=2)
		nnArgs.add_argument('--nTurn', type=int, default=3)
		nnArgs.add_argument('--speakerEmbedSize', type=int, default=0)
		nnArgs.add_argument('--nLSTM', type=int, default=3, help='in DCRNN, this is the ')
		nnArgs.add_argument('--heads', type=int, default=3)
		nnArgs.add_argument('--attn', action='store_true')
		nnArgs.add_argument('--selfattn', action='store_true')
		nnArgs.add_argument('--nContexts', type=int, default=4)
		nnArgs.add_argument('--independent', action='store_true')

		# training options
		trainingArgs = parser.add_argument_group('Training options')
		trainingArgs.add_argument('--model', type=str, help='hbmp, dcrnn, transfer+hbmp, transfer+dcrnn, hbmp+share')
		trainingArgs.add_argument('--eager', action='store_true', help='turn on eager mode for debugging')
		trainingArgs.add_argument('--modelPath', type=str, default='saved')
		trainingArgs.add_argument('--preEmbedding', action='store_true')
		trainingArgs.add_argument('--elmo', action='store_true')
		trainingArgs.add_argument('--trainElmo', action='store_true')
		trainingArgs.add_argument('--dropOut', type=float, default=1.0, help='dropout rate for RNN (keep prob)')
		trainingArgs.add_argument('--learningRate', type=float, default=0.001, help='learning rate')
		trainingArgs.add_argument('--batchSize', type=int, default=100, help='batch size')
		trainingArgs.add_argument('--epochs', type=int, default=200, help='most training epochs')
		trainingArgs.add_argument('--device', type=str, default='/gpu:0', help='use the first GPU as default')
		trainingArgs.add_argument('--loadModel', action='store_true', help='whether or not to use old models')
		trainingArgs.add_argument('--sampleWeight', default=6.848, type=float, help='a constant to balance different categories')
		trainingArgs.add_argument('--weighted', action='store_true', help='whether or not to weight the training samples')
		trainingArgs.add_argument('--ffnn', type=int, default=500, help='intermediate ffnn size')
		trainingArgs.add_argument('--gamma', type=float, default=0.1, help='we use a lambda to balance between snli and emo,'
		                                                        'multiply gradients snli samples by this lambda')
		trainingArgs.add_argument('--universal', action='store_true', help='whether or not to use universal sent embeddings')
		trainingArgs.add_argument('--augment', action='store_true', help='data augumentation by adding random 2nd sentences')

		"""
		in training data: happy, sad, angry: 5k (16.67%) each; others: 15k (50%)
		in dev/test: happy, sad, angry: 4% each; others: 88%
					88/3 = 29.333
					29.33/4 = 7.33

in genuine data:					
					
30160 train, happy = 4243, of 0.14068302387267906, sad = 5463, of 0.1811339522546419, angry = 5506, of 0.18255968169761272, others = 14948, of 0.4956233421750663
2755 val, happy = 142, of 0.051542649727767696, sad = 125, of 0.045372050816696916, angry = 150, of 0.0544464609800363, others = 2338, of 0.8486388384754991
		
		"""
		return parser.parse_args(args)

	def statistics(self):
		"""
		27144 train, happy = 3815, of 0.14054671382257589, sad = 4920, of 0.1812555260831123,
		            angry = 4977, of 0.1833554376657825, others = 13432, of 0.4948423224285293

		3016 val, happy = 428, of 0.1419098143236074, sad = 543, of 0.18003978779840848,
					angry = 529, of 0.17539787798408488, others = 1516, of 0.5026525198938993
		:return:
		"""
		train_samples = self.textData.train_samples
		val_samples = self.textData.valid_samples

		def cnt(samples, tag='train'):
			happy = 0
			sad = 0
			angry = 0
			others = 0
			for sample in samples:
				if sample.label == self.textData.label2idx['happy']:
					happy += 1
				elif sample.label == self.textData.label2idx['sad']:
					sad += 1
				elif sample.label == self.textData.label2idx['angry']:
					angry += 1
				else:
					others += 1
			total = happy + sad + angry + others
			print('{} {}, happy = {}, of {}, sad = {}, of {}, angry = {}, of {}, others = {}, of {}'
			      .format(total, tag, happy, happy/total, sad, sad/total, angry, angry/total, others, others/total))

		cnt(train_samples, 'train')
		cnt(val_samples, 'val')
		exit(0)


	def main(self, args=None):
		print('TensorFlow version {}'.format(tf.VERSION))

		# initialize args
		self.args = self.parse_args(args)

		self.resultDir = os.path.join(self.args.resultDir, self.args.dataset)
		self.summaryDir = os.path.join(self.args.summaryDir, self.args.dataset)
		self.dataDir = os.path.join(self.args.dataDir, self.args.dataset)
		self.testDir = os.path.join(self.args.testDir, self.args.dataset)

		self.outFile = utils.constructFileName(self.args, prefix=self.resultDir)
		self.testFile = utils.constructFileName(self.args, prefix=self.testDir)

		self.args.datasetName = utils.constructFileName(self.args, prefix=self.args.dataset, createDataSetName=True)
		datasetFileName = os.path.join(self.dataDir, self.args.datasetName)

		if not os.path.exists(self.resultDir):
			os.makedirs(self.resultDir)

		if not os.path.exists(self.testDir):
			os.makedirs(self.testDir)

		if not os.path.exists(self.args.modelPath):
			os.makedirs(self.args.modelPath)

		if not os.path.exists(self.summaryDir):
			os.makedirs(self.summaryDir)

		if not os.path.exists(datasetFileName):
			self.textData = TextData(self.args)
			with open(datasetFileName, 'wb') as datasetFile:
				p.dump(self.textData, datasetFile)
			print('dataset created and saved to {}, exiting ...'.format(datasetFileName))
			exit(0)
		else:
			with open(datasetFileName, 'rb') as datasetFile:
				self.textData = p.load(datasetFile)
			print('dataset loaded from {}'.format(datasetFileName))

		# self.statistics()

		sessConfig = tf.ConfigProto(allow_soft_placement=True)
		sessConfig.gpu_options.allow_growth = True

		self.model_path = os.path.join(self.args.modelPath, self.args.dataset)
		self.model_path = utils.constructFileName(self.args, prefix=self.model_path, tag='model')
		if not os.path.exists(self.model_path):
			os.makedirs(self.model_path)
		self.model_name = os.path.join(self.model_path, 'model')

		self.sess = tf.Session(config=sessConfig)
		# summary writer
		self.summaryDir = utils.constructFileName(self.args, prefix=self.summaryDir)
		if self.args.eager:
			tf.enable_eager_execution(config=sessConfig, device_policy=tf.contrib.eager.DEVICE_PLACEMENT_WARN)
			print('eager execution enabled')

		# import timeit
		#
		# start = timeit.default_timer()
		#
		# self.textData.get_batches(tag='train', augment=self.args.augment)
		#
		# stop = timeit.default_timer()
		#
		# print('Time: ', stop - start)
		# exit(0)

		with tf.device(self.args.device):
			if self.args.model.find('hbmp') != -1:
				if self.args.model.find('share') != -1:
					print('Creating model with HBMP share')
					self.model = ModelHBMPShare(self.args, self.textData, eager=self.args.eager)
				else:
					print('Creating model with HBMP ordinary')
					self.model = ModelHBMP(self.args, self.textData, eager=self.args.eager)
			else:
				self.model = ModelBasic(self.args, self.textData, eager=self.args.eager)
				print('Basic model created!')

			if self.args.eager:
				self.train_eager()
				exit(0)
			# saver can only be created after we have the model
			self.saver = tf.train.Saver()

			self.summaryWriter = tf.summary.FileWriter(self.summaryDir, self.sess.graph)
			self.mergedSummary = tf.summary.merge_all()

			if self.args.loadModel:
				# load model from disk
				if not os.path.exists(self.model_path):
					print('model does not exist on disk!')
					print(self.model_path)
					exit(-1)

				self.saver.restore(sess=self.sess, save_path=self.model_name)
				print('Variables loaded from disk {}'.format(self.model_name))
			else:
				init = tf.global_variables_initializer()
				table_init = tf.tables_initializer()
				# initialize all global variables
				self.sess.run([init, table_init])
				print('All variables initialized')

			self.train(self.sess)

	def train_eager(self):
		for e in range(self.args.epochs):
			trainBatches = self.textData.train_batches

			for idx, nextBatch in enumerate(tqdm(trainBatches)):
				self.model.step(nextBatch, test=False, eager=self.args.eager)
				self.model.buildNetwork()

				print()

	def train(self, sess):
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)

		print('Start training')

		out = open(self.outFile, 'w', 1)
		out.write(self.outFile + '\n')
		utils.writeInfo(out, self.args)

		current_val_f1_micro_unweighted = 0.0

		for e in range(self.args.epochs):
			# training
			trainBatches = self.textData.get_batches(tag='train', augment=self.args.augment)
			#trainBatches = self.textData.train_batches
			totalTrainLoss = 0.0

			# cnt of batches
			cnt = 0

			total_samples = 0
			total_corrects = 0

			all_predictions = []
			all_labels = []
			all_sample_weights = []

			for idx, nextBatch in enumerate(tqdm(trainBatches)):

				cnt += 1
				self.globalStep += 1
				total_samples += nextBatch.batch_size
				# print(idx)

				ops, feed_dict, labels, sample_weights = self.model.step(nextBatch, test=False)
				_, loss, predictions, corrects = sess.run(ops, feed_dict)
				all_predictions.extend(predictions)
				all_labels.extend(labels)
				all_sample_weights.extend(sample_weights)
				total_corrects += corrects
				totalTrainLoss += loss

				self.summaryWriter.add_summary(utils.makeSummary({"train_loss": loss}), self.globalStep)
				#break
			trainAcc = total_corrects * 1.0 / total_samples

			# calculate f1 score for train (weighted/unweighted)
			train_f1_micro, train_f1_macro, train_p_micro, train_r_micro, train_p_macro, train_r_macro\
				= self.cal_F1(y_pred=all_predictions, y_true=all_labels)
			train_f1_micro_w, train_f1_macro_w, train_p_micro_w, train_r_micro_w, train_p_macro_w, train_r_macro_w\
				= self.cal_F1(y_pred=all_predictions, y_true=all_labels,
			                                                 sample_weight=all_sample_weights)

			print('\nepoch = {}, Train, loss = {}, trainAcc = {}, train_f1_micro = {}, train_f1_macro = {},'
			      ' train_f1_micro_w = {}, train_f1_macro_w = {}'.
			      format(e, totalTrainLoss, trainAcc, train_f1_micro, train_f1_macro, train_f1_micro_w, train_f1_macro_w))
			print('\ttrain_p_micro = {}, train_r_micro = {}, train_p_macro = {}, train_r_macro = {}'.format(train_p_micro, train_r_micro, train_p_macro, train_r_macro))
			print('\ttrain_p_micro_w = {}, train_r_micro_w = {}, train_p_macro_w = {}, train_r_macro_w = {}'.format(train_p_micro_w, train_r_micro_w, train_p_macro_w, train_r_macro_w))


			out.write('\nepoch = {}, loss = {}, trainAcc = {}, train_f1_micro = {}, train_f1_macro = {},'
			          ' train_f1_micro_w = {}, train_f1_macro_w = {}\n'.
			          format(e, totalTrainLoss, trainAcc, train_f1_micro, train_f1_macro, train_f1_micro_w, train_f1_macro_w))
			out.write('\ttrain_p_micro = {}, train_r_micro = {}, train_p_macro = {}, train_r_macro = {}\n'.format(train_p_micro, train_r_micro, train_p_macro, train_r_macro))
			out.write('\ttrain_p_micro_w = {}, train_r_micro_w = {}, train_p_macro_w = {}, train_r_macro_w = {}\n'.format(train_p_micro_w, train_r_micro_w, train_p_macro_w, train_r_macro_w))
			#continue

			out.flush()

			# calculate f1 score for val (weighted/unweighted)
			valAcc, valLoss, val_f1_micro, val_f1_macro, val_f1_micro_w, val_f1_macro_w,\
			val_p_micro, val_r_micro, val_p_macro, val_r_macro, val_p_micro_w, val_r_micro_w, val_p_macro_w, val_r_macro_w\
				= self.test(sess, tag='val')

			print('\n\tVal, loss = {}, valAcc = {}, val_f1_micro = {}, val_f1_macro = {}, val_f1_micro_w = {}, val_f1_macro_w = {}'.
			      format(valLoss, valAcc, val_f1_micro, val_f1_macro, val_f1_micro_w, val_f1_macro_w))
			print('\t\t val_p_micro = {}, val_r_micro = {}, val_p_macro = {}, val_r_macro = {}'.format(val_p_micro, val_r_micro, val_p_macro, val_r_macro))
			print('\t\t val_p_micro_w = {}, val_r_micro_w = {}, val_p_macro_w = {}, val_r_macro_w = {}'.format(val_p_micro_w, val_r_micro_w, val_p_macro_w, val_r_macro_w))

			out.write('\n\tVal, loss = {}, valAcc = {}, val_f1_micro = {}, val_f1_macro = {}, val_f1_micro_w = {}, val_f1_macro_w = {}\n'.
			      format(valLoss, valAcc, val_f1_micro, val_f1_macro, val_f1_micro_w, val_f1_macro_w))
			out.write('\t\t val_p_micro = {}, val_r_micro = {}, val_p_macro = {}, val_r_macro = {}\n'.format(val_p_micro, val_r_micro, val_p_macro, val_r_macro))
			out.write('\t\t val_p_micro_w = {}, val_r_micro_w = {}, val_p_macro_w = {}, val_r_macro_w = {}\n'.format(val_p_micro_w, val_r_micro_w, val_p_macro_w, val_r_macro_w))


			# calculate f1 score for test (weighted/unweighted)
			testAcc, testLoss, test_f1_micro, test_f1_macro, test_f1_micro_w, test_f1_macro_w,\
			test_p_micro, test_r_micro, test_p_macro, test_r_macro, test_p_micro_w, test_r_micro_w, test_p_macro_w, test_r_macro_w\
				= self.test(sess, tag='test')

			print('\n\ttest, loss = {}, testAcc = {}, test_f1_micro = {}, test_f1_macro = {}, test_f1_micro_w = {}, test_f1_macro_w = {}'.
			      format(testLoss, testAcc, test_f1_micro, test_f1_macro, test_f1_micro_w, test_f1_macro_w))
			print('\t\t test_p_micro = {}, test_r_micro = {}, test_p_macro = {}, test_r_macro = {}'.format(test_p_micro, test_r_micro, test_p_macro, test_r_macro))
			print('\t\t test_p_micro_w = {}, test_r_micro_w = {}, test_p_macro_w = {}, test_r_macro_w = {}'.format(test_p_micro_w, test_r_micro_w, test_p_macro_w, test_r_macro_w))

			out.write('\n\ttest, loss = {}, testAcc = {}, test_f1_micro = {}, test_f1_macro = {}, test_f1_micro_w = {}, test_f1_macro_w = {}\n'.
			      format(testLoss, testAcc, test_f1_micro, test_f1_macro, test_f1_micro_w, test_f1_macro_w))
			out.write('\t\t test_p_micro = {}, test_r_micro = {}, test_p_macro = {}, test_r_macro = {}\n'.format(test_p_micro, test_r_micro, test_p_macro, test_r_macro))
			out.write('\t\t test_p_micro_w = {}, test_r_micro_w = {}, test_p_macro_w = {}, test_r_macro_w = {}\n'.format(test_p_micro_w, test_r_micro_w, test_p_macro_w, test_r_macro_w))


			out.flush()

			self.summaryWriter.add_summary(utils.makeSummary({"train_acc": trainAcc}), e)
			self.summaryWriter.add_summary(utils.makeSummary({"val_acc": valAcc}), e)

			# use val_f1_micro (unweighted) as metric
			if val_f1_micro >= current_val_f1_micro_unweighted:
				current_val_f1_micro_unweighted = val_f1_micro
				print('New val f1_micro {} at epoch {}'.format(val_f1_micro, e))
				out.write('New val f1_micro {} at epoch {}\n'.format(val_f1_micro, e))

				save_path = self.saver.save(sess, save_path=self.model_name)
				print('model saved at {}'.format(save_path))
				out.write('model saved at {}\n'.format(save_path))

				test_predictions = self.test(sess, tag='test2')
				print('Writing predictions at epoch {}'.format(e))
				out.write('Writing predictions at epoch {}\n'.format(e))
				test_file = self.write_predictions(test_predictions, tag='unweighted')

				print('Writing predictions to {}'.format(test_file))
				out.write('Writing predictions to {}\n'.format(test_file))

			out.flush()
		out.close()


	def write_predictions(self, predictions, tag='weighted'):
		test_file = self.testFile + '_' + tag
		with open(test_file, 'w') as file:
			file.write('id\tturn1\tturn2\tturn3\tlabel\n')
			idx2label = {v: k for k, v in self.textData.label2idx.items()}
			for idx, sample in enumerate(self.textData.test_samples):
				assert idx == sample.id

				file.write(str(idx)+'\t')
				for ind, sent in enumerate(sample.sents):
					file.write(' '.join(sent[:sample.length[ind]]).encode('ascii', 'ignore').decode('ascii')+'\t')
				file.write(idx2label[predictions[idx]]+'\n')
		return test_file

	def test(self, sess, tag='val'):
		"""
		for the real dev data, during test, do not use sample weights
		:param sess:
		:param tag:
		:return:
		"""
		if tag == 'val':
			print('Validating\n')
			batches = self.textData.val_batches
		else:
			print('Testing\n')
			batches = self.textData.test_batches

		cnt = 0

		total_samples = 0
		total_corrects = 0
		total_loss = 0.0
		all_predictions = []
		all_labels = []
		all_sample_weights = []
		for idx, nextBatch in enumerate(tqdm(batches)):
			cnt += 1

			total_samples += nextBatch.batch_size
			ops, feed_dict, labels, sample_weights = self.model.step(nextBatch, test=True)

			loss, predictions, corrects = sess.run(ops, feed_dict)
			all_predictions.extend(predictions)
			all_labels.extend(labels)
			all_sample_weights.extend(sample_weights)
			total_loss += loss
			total_corrects += corrects

			#break

		f1_micro, f1_macro, p_micro, r_micro, p_macro, r_macro = self.cal_F1(y_pred=all_predictions, y_true=all_labels)
		f1_micro_w, f1_macro_w, p_micro_w, r_micro_w, p_macro_w, r_macro_w =\
			self.cal_F1(y_pred=all_predictions, y_true=all_labels, sample_weight=all_sample_weights)

		acc = total_corrects * 1.0 / total_samples

		if tag == 'test2':
			return all_predictions
		else:
			return acc, total_loss, f1_micro, f1_macro, f1_micro_w, f1_macro_w,\
			       p_micro, r_micro, p_macro, r_macro,\
			       p_micro_w, r_micro_w, p_macro_w, r_macro_w

	def cal_F1(self, y_pred, y_true, sample_weight=None):
		labels = [self.textData.label2idx['happy'],
		          self.textData.label2idx['sad'],
		          self.textData.label2idx['angry']]
		# if sample_weight is not None:
		# 	sample_weight = np.asarray(sample_weight)
		# 	sample_weight = sample_weight.astype(int)
		p_micro, r_micro, f1_micro, _ = \
			precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro', labels=labels, sample_weight=sample_weight)
		p_macro, r_macro, f1_macro, _ = \
			precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro', labels=labels, sample_weight=sample_weight)

		return f1_micro, f1_macro, p_micro, r_micro, p_macro, r_macro
