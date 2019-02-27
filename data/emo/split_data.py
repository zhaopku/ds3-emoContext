import random

ratio = 0.9

def write_to_disk(samples, tag='train'):
	with open(tag+'_splited.txt', 'w') as file:
		for sample in samples:
			sample = sample.strip()
			file.write(sample+'\n')

with open('train.txt', 'r') as file:
	lines = file.readlines()
	samples = lines[1:]
	random.seed(623)
	random.shuffle(samples)

	n_samples = len(samples)

	print('Number of samples = {}'.format(n_samples))

	n_train_splited = int(ratio*n_samples)
	n_val_splited = n_samples - n_train_splited

	print('Number of training samples = {}'.format(n_train_splited))
	print('Number of val samples = {}'.format(n_val_splited))

	train_samples = samples[:n_train_splited]
	val_samples = samples[n_train_splited:]

	write_to_disk(train_samples, tag='train')
	write_to_disk(val_samples, tag='val')


