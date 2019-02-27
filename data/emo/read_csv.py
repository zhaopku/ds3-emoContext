import csv

with open('train_liwc.csv', 'r') as file:
	reader = csv.reader(file)

	for row in reader:
		print()
		print(row[-93:])