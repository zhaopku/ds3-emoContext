class Sample:
	# def __init__(self, data, words, steps, label, length):
	#     self.input_ = data[0:steps]
	#     self.sentence = words[0:steps]
	#     self.length = length
	#     self.label = label

	def __init__(self):
		self.label = -1

		# list of lists
		self.sents = [[]]*3
		self.data = [[]]*3

		# [3, 93]
		self.liwc_features = []

		# true length of data
		self.length = [-1]*3

		self.id = -1
		self.sample_weight = 0.0

	def set_sent(self, sent, data, tag, length):
		self.sents[tag] = sent
		self.data[tag] = data
		self.length[tag] = length

	def set_label(self, label, sample_weight):
		self.label = label
		self.sample_weight = sample_weight

	def set_id(self, id):
		self.id = id

class Batch:
	def __init__(self, samples):
		self.samples = samples
		self.batch_size = len(samples)
