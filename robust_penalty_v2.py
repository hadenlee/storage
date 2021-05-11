import numpy as np
import pandas as pd
import scipy.special as ss

import seaborn as sns
import matplotlib.pyplot as plt


# https://www.notion.so/hadenlee/Mechanisms-for-Storage-35fd80df1ce141fda7f57428dd7f5b6d#3cfe8762ea844b06be81106167ef7ba3
# d \cdot \delta_S  \ge (e^{F\cdot\beta\cdot z + \ln(\delta_S)}) / (z \cdot F) - v

def get_min_d(ff, beta, pe, ds, v):
  z = -np.log(pe)
  return np.exp(ff * beta * z) * ds / (z * ff) - v

# https://www.notion.so/hadenlee/Mechanisms-for-Storage-35fd80df1ce141fda7f57428dd7f5b6d#93327f01216b400b8bb2fdda44f68f98
# d \cdot \delta_S \le (\epsilon_C \cdot (A'+1)/F - v + \delta_S) \cdot (-1/p_\epsilon) - v
def get_max_d(ff, aa, ec, v, ds, pe):
  return (((aa + 1) / ff) * ec - v + ds) * (-1/pe) - v
  # return (((aa + 1) / ff) * ec - v + ds) * (-1/pe) - v


  
def gen_csv_data():
  A = [i for i in range(6, 101, 3)]
  F = [i for i in range(2, 35)]
  v = 1
  PE = [i/200 for i in range(1, 101, 1)]
  B = [i/200 for i in range(1, 201, 1)]
  DS = [0.001]
  E = [0.001]

  fname = "data/data_range_on_d_v2.csv"
  print(fname)

  data = []
  for beta in B:
    print(beta)
    for ds in DS:
      for pe in PE:
        for ec in E:
          for aa in A:
            for ff in F:
              if ff > aa/3:
                continue
            
              min_d = get_min_d(ff, beta, pe, ds, v)
              max_d = get_max_d(ff, aa, ec, v, ds, pe)
          
              data.append([aa, ff, pe, ds, ec, beta, min_d, max_d])


  df = pd.DataFrame(data, columns = ['A', 'F', 'pe', 'ds', 'ec', 'beta', 'min_d', 'max_d'])
  print(df)

  df.to_csv(fname, sep=',')

# -----------------------------------------------------
print('hello')

gen_csv_data()
