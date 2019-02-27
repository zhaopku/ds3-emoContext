import jsonlines

file_name = ['devwithoutlabels.txt', 'train_splited.txt', 'val_splited.txt']

def convert(path):
	with open(path, 'r') as in_file, open(path+'.out.txt', 'w') as out_file:
		lines = in_file.readlines()
		for line in lines:
			if path.find('dev') == -1:
				sents = line.split('\t')[1:-1]
			else:
				sents = line.split('\t')[1:]
			for sent in sents:
				out_file.write(sent.strip()+'\n')

for f in file_name:
	convert(f)