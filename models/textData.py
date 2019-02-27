import os
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm
from models.data_utils import Sample, Batch
import nltk
import csv

class TextData:
	def __init__(self, args):
		self.args = args

		self.UNK_WORD = 'unk'
		self.PAD_WORD = '<pad>'

		self.label2idx = {'happy':0,
		                  'sad':1,
		                  'angry':2,
		                  'others':3}

		self.idx2label = {v: k for k, v in self.label2idx.items()}

		# list of batches
		self.train_batches = []
		self.val_batches = []
		self.test_batches = []

		self.word2id = {}
		self.id2word = {}

		self.train_samples = None
		self.valid_samples = None
		self.test_samples = None
		self.preTrainedEmbedding = None

		self.train_samples, self.valid_samples, self.test_samples = self._create_data()
		self.happy_samples, self.sad_samples, self.angry_samples, self.others_samples = self.split_training_samples(self.train_samples)

		# [num_batch, batch_size, maxStep]
		self.train_batches = self._create_batch(self.train_samples)
		self.val_batches = self._create_batch(self.valid_samples)

		# note: test_batches is none here
		self.test_batches = self._create_batch(self.test_samples)

	@staticmethod
	def split_training_samples(all_samples):
		happy_samples = []
		sad_samples = []
		angry_samples = []
		others_samples = []
		for sample in all_samples:
			# happy
			if sample.label == 0:
				happy_samples.append(sample)
			elif sample.label == 1:
				sad_samples.append(sample)
			elif sample.label == 2:
				angry_samples.append(sample)
			elif sample.label == 3:
				others_samples.append(sample)
			else:
				print('illegal sample label')
				exit(-1)

		return happy_samples, sad_samples, angry_samples, others_samples

	def getVocabularySize(self):
		assert len(self.word2id) == len(self.id2word)
		return len(self.word2id)

	@staticmethod
	def reconstruct(samples):
		new_idx = np.arange(len(samples))
		np.random.shuffle(new_idx)
		mid_sent = []
		mid_length = []
		mid_data = []

		for sample in samples:

			mid_sent.append(sample.sents[1])
			mid_length.append(sample.length[1])
			mid_data.append(sample.data[1])

		for idx, sample in enumerate(samples):
			sample.sents[1] = mid_sent[new_idx[idx]]
			sample.length[1] = mid_length[new_idx[idx]]
			sample.data[1] = mid_data[new_idx[idx]]

		return samples


	def _create_batch(self, all_samples, tag='test', augment=False):
		all_batches = []
		if tag == 'train':

			if augment:
				happy_samples = self.reconstruct(self.happy_samples)
				sad_samples = self.reconstruct(self.sad_samples)
				angry_samples = self.reconstruct(self.angry_samples)
				others_samples = self.reconstruct(self.others_samples)

				all_samples = happy_samples + sad_samples + angry_samples + others_samples

			random.shuffle(all_samples)

		if all_samples is None:
			return all_batches

		num_batch = len(all_samples)//self.args.batchSize + 1
		for i in range(num_batch):
			samples = all_samples[i*self.args.batchSize:(i+1)*self.args.batchSize]

			if len(samples) == 0:
				continue

			batch = Batch(samples)
			all_batches.append(batch)

		return all_batches

	def _create_samples(self, file_path, tag):

		oov_cnt = 0
		cnt = 0
		with open(file_path, 'r') as file:
			lines = file.readlines()
			all_samples = []
			for idx, line in enumerate(tqdm(lines)):
				# if idx == 100000:
				#     break
				line = line.strip()
				splits = line.split('\t')

				sample = Sample()
				if tag != 'testxs':
					label = self.label2idx[splits[-1]]
					if label == self.label2idx['happy']:
						sample_weight = 1.464
					elif label == self.label2idx['sad']:
						sample_weight = 1.0
					elif label == self.label2idx['angry']:
						sample_weight = 1.192
					elif label == self.label2idx['others']:
						sample_weight = 6.848


					sample.set_label(label=label, sample_weight=sample_weight)
				else:
					# use dummy label for test data
					sample.set_label(label=0, sample_weight=1.0)
				if len(splits[0]) < 6:
					sample.set_id(id=int(splits[0]))
				else:
					sample.set_id(id=10000000)
				sents = splits[1:4]
				for j, sent in enumerate(sents):
					words = nltk.word_tokenize(sent)
					word_ids = []

					length = len(words)
					cnt += length
					for word in words:
						if word in self.word2id.keys():
							id_ = self.word2id[word]
						else:
							id_ = self.word2id[self.UNK_WORD]
							print('Check!')
							exit(-1)
						if id_ == self.word2id[self.UNK_WORD] and word != self.UNK_WORD:
							oov_cnt += 1
						word_ids.append(id_)

					while len(word_ids) < self.args.maxSteps:
						word_ids.append(self.word2id[self.PAD_WORD])

					length_train = len(words[:self.args.maxSteps])
					while len(words) < self.args.maxSteps:
						words.append(self.PAD_WORD)

					sample.set_sent(sent=words[:self.args.maxSteps], data=word_ids[:self.args.maxSteps], tag=j, length = length_train)

				all_samples.append(sample)

		return all_samples, oov_cnt, cnt

	def create_embeddings(self):
		words = self.word2id.keys()

		glove_embed = {}

		with open(self.args.embeddingFile, 'r') as glove:
			lines = glove.readlines()
			for line in tqdm(lines, desc='glove'):
				splits = line.split()
				word = splits[0]
				if len(splits) > 301:
					word = ''.join(splits[0:len(splits) - 300])
					splits[1:] = splits[len(splits) - 300:]
				if word not in words:
					continue
				embed = [float(s) for s in splits[1:]]
				glove_embed[word] = embed

		embeds = []
		for word_id in range(len(self.id2word)):
			word = self.id2word[word_id]
			if word in glove_embed.keys():
				embed = glove_embed[word]
			else:
				embed = glove_embed[self.UNK_WORD]
				self.word2id[word] = self.word2id[self.UNK_WORD]
			embeds.append(embed)

		embeds = np.asarray(embeds)

		return embeds

	def _create_data(self):

		train_path = os.path.join(self.args.dataDir, self.args.dataset, self.args.trainFile)
		val_path = os.path.join(self.args.dataDir, self.args.dataset, self.args.valFile)
		test_path = os.path.join(self.args.dataDir, self.args.dataset, self.args.testFile)

		print('Building vocabularies for {} dataset'.format(self.args.dataset))
		self.word2id, self.id2word = self._build_vocab(train_path, val_path, test_path)

		print('Creating pretrained embeddings!')
		self.preTrainedEmbedding = self.create_embeddings()

		print('Building training samples!')
		train_samples, train_oov, train_cnt = self._create_samples(train_path, tag='train')
		val_samples, val_oov, val_cnt = self._create_samples(val_path, tag='val')
		test_samples, test_oov, test_cnt = self._create_samples(test_path, tag='test')

		print('OOV rate for train = {:.2%}'.format(train_oov*1.0/train_cnt))
		print('OOV rate for val = {:.2%}'.format(val_oov*1.0/val_cnt))
		print('OOV rate for test = {:.2%}'.format(test_oov*1.0/test_cnt))

		return train_samples, val_samples, test_samples

	def _read_sents(self, filename, tag):
		with open(filename, 'r') as file:
			all_words = []
			lines = file.readlines()
			all_length = []
			for idx, line in enumerate(tqdm(lines)):
				line = line.strip()

				splits = line.split('\t')
				try:
					assert (len(splits) == 5 and (tag == 'train' or tag == 'val')) or \
					       (len(splits) == 5 and tag == 'test')
				except:
					print('id = {}'.format(splits[0]))
					exit(-1)


				for sent in splits[1:4]:
					words = nltk.word_tokenize(sent)
					all_length.append(len(words))
					all_words.extend(words)

			assert len(all_length) == 3*len(lines)
			n_out_of_steps = np.sum(np.asarray(all_length) <= self.args.maxSteps)

			print('{} % of samples <= {}'.format(n_out_of_steps*100/len(all_length), self.args.maxSteps))
			print('Avg {} length = {}, max length = {}'.format(tag, np.average(all_length), np.max(all_length)))
		return all_words

	def _build_vocab(self, train_path, val_path, test_path):

		all_train_words = self._read_sents(train_path, tag='train')
		all_val_words = self._read_sents(val_path, tag='val')
		all_test_words = self._read_sents(test_path, tag='test')

		all_words = all_train_words + all_val_words + all_test_words

		print(len(list(set(all_words))))

		counter = Counter(all_words)

		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		# keep the most frequent vocabSize words, including the special tokens
		# -1 means we have no limits on the number of words
		if self.args.vocabSize != -1:
			count_pairs = count_pairs[0:self.args.vocabSize-2]

		count_pairs.append((self.UNK_WORD, 100000))
		count_pairs.append((self.PAD_WORD, 100000))

		if self.args.vocabSize != -1:
			assert len(count_pairs) == self.args.vocabSize

		words, _ = list(zip(*count_pairs))
		word_to_id = dict(zip(words, range(len(words))))

		id_to_word = {v: k for k, v in word_to_id.items()}

		return word_to_id, id_to_word

	def get_batches(self, tag='train', augment=False):
		if tag == 'train':
			return self._create_batch(self.train_samples, tag='train', augment=augment)
		elif tag == 'val':
			return self.val_batches
		else:
			return self.test_batches
