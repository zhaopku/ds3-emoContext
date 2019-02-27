in_files = ['train.txt.cleaned.txt', 'train_new.txt']

out_file = 'train_big.txt'

with open(in_files[0], 'r') as in_0, open(in_files[1], 'r') as in_1, open(out_file, 'w') as out:
	lines_0 = in_0.readlines()
	lines_1 = in_1.readlines()

	lines = lines_0 + lines_1

	for line in lines:
		out.write(line)