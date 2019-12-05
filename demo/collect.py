import numpy as np

accs = []
with open("res/res.tsv", 'rt') as f:
	for l in f:
		word, acc = l.split()
		accs.append(float(acc))

print(np.mean(accs))
