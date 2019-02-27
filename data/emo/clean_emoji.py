import emoji
from tqdm import tqdm

file_names = ['test.txt']
import nltk

class Sample:
	# def __init__(self, data, words, steps, label, length):
	#     self.input_ = data[0:steps]
	#     self.sentence = words[0:steps]
	#     self.length = length
	#     self.label = label

	def __init__(self):
		self.label = ''

		# list of lists
		self.sents = [[]]*3
		self.id = -1


def convert(file_name, test=False):
	with open(file_name, 'r') as file:
		lines = file.readlines()
		all_samples = []

		for idx, line in enumerate(tqdm(lines, desc=file_name)):
			sample = Sample()
			line = line.strip()
			splits = line.split('\t')
			sample.id = int(splits[0])
			sents = splits[1:4]
			for j, sent in enumerate(sents):
				words = nltk.word_tokenize(sent)
				new_words = []
				for word in words:
					new_word = emoji.demojize(word, delimiters=('', '')).split('_')
					new_word = [x.strip() for x in new_word]
					new_words.extend(new_word)

				sample.sents[j] = new_words

			sample.label = splits[-1].strip()
			all_samples.append(sample)


		with open(file_name+'.cleaned.txt', 'w') as file:
			for idx, sample in enumerate(all_samples):
				assert idx == sample.id

				file.write(str(idx)+'\t')
				for ind, sent in enumerate(sample.sents):
					assert len(sent) >= 1
					file.write(' '.join(sent)+'\t')
				if not test:
					file.write(sample.label+'\n')
				else:
					file.write('\n')


for f in file_names:
	if f.startswith('xxx'):
		convert(f, test=True)
	else:
		convert(f)