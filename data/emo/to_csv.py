import csv
import nltk
from tqdm import tqdm

file_names = ['train.txt', 'dev.txt']

def write_to_csv(file_name):
	with open(file_name+'.csv', 'w') as f_out, open(file_name, 'r') as f_in:
		lines = f_in.readlines()
		writer = csv.writer(f_out)
		for line in tqdm(lines, desc=file_name):
			sents = line.split('\t')[1:-1]

			for sent in sents:
				words = nltk.word_tokenize(sent.strip())
				writer.writerow(words)


for file_name in file_names:
	write_to_csv(file_name)

