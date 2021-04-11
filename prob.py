import numpy as np
import pandas as pd
import seaborn as sns
import scipy.special as ss
import matplotlib.pyplot as plt

def compute_prob(a_i, aa, x, ff):
  return ss.binom(a_i, x) * ss.binom(aa, ff-x) / ss.binom(aa+a_i, ff)

for ff in range(5, 6):
	for aa in range(10, 11):
		for a_i in range(ff, aa+1):
			v1 = [compute_prob(a_i, aa, x, ff) for x in range(0, ff+1)]
			v2 = [compute_prob(a_i+1, aa, x, ff) for x in range(0, ff+1)]
			print('ff = %d aa = %d a_i = %d' % (ff, aa, a_i))
			x1, x2 = 0, 0
			diff = []
			for i in range(len(v1)):
				x1 += i * v1[i]
				x2 += i * v2[i]
				diff.append(v2[i] - v1[i])
			print(v1, sum(v1), x1)
			print(v2, sum(v2), x2)
			print(diff)
			print('')
