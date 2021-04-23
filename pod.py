import numpy as np
import pandas as pd
import scipy.special as ss

import seaborn as sns
import matplotlib.pyplot as plt

def save_plot(d_sub, title, col_x, col_y, col_hue, col_col, col_wrap):
  sns.set_style("whitegrid")

  g = sns.catplot(x=col_x, y=col_y, hue=col_hue, col=col_col,
                  capsize=.2, palette="tab10", 
                  height=6, aspect=.75,
                  col_wrap=col_wrap,
                  kind="point", data=d_sub)
  g.despine(left=True)
  g.savefig(title)


def plots_Ti_1(df, prefix):
  print('obtaining a subset of data')
  d_sub = (df[(df['p_e'] == 0.01) & (df['d'] == 21) & (df['A']%6==0) & (df['F']%4 == 1)])[["T_i", "rev", "F", "A"]]
  print(d_sub)
  print(d_sub.info())

  title = '%s_pe1_d21.png' % (prefix)
  save_plot(d_sub, title, "T_i", "rev", "F", "A", 4)


def plots_Ti_2(df, prefix):
  print('obtaining a subset of data')
  d_sub = (df[(df['p_e'] == 0.01) & (df['d'] == 21) & (df['A']%6==0) & (df['F']%4 == 1)])[["T_i", "rev", "F", "A"]]
  print(d_sub)
  print(d_sub.info())

  title = '%s_pe1_d21.png' % (prefix)
  save_plot(d_sub, title, "T_i", "rev", "A", "F", 4)


def plots_Ti_3(df, prefix):
  print('obtaining a subset of data')
  d_sub = (df[(df['d'] > 1) & (df['A']==30) & (df['F'] == 10)])[["T_i", "rev", "p_e", "d"]]
  print(d_sub)
  print(d_sub.info())

  title = '%s_A30_F10.png' % (prefix)
  save_plot(d_sub, title, "T_i", "rev", "p_e", "d", 4)

def plots_Ti_4(df, prefix):
  print('obtaining a subset of data')
  d_sub = (df[(df['d'] > 1) & (df['A']==30) & (df['F'] == 10)])[["T_i", "rev", "p_e", "d"]]
  print(d_sub)
  print(d_sub.info())

  title = '%s_A30_F10.png' % (prefix)
  save_plot(d_sub, title, "T_i", "rev", "d", "p_e", 3)


def lower(b, ff, v, pe, ds):
  z = -np.log(pe)
  return ds * np.exp(ff * b * z) / (z * ff) - v
  
def upper(aa, ff, v, ds, pe, ec):
  return (ec * (aa+1)/ ff - v + ds) * (-1 / pe) - v

def max_beta(aa, ff, v, ds, pe, ec):
  z = -np.log(pe)
  return (np.log(z * ff * (ec * (aa+1) / ff - v + ds) / (-1. / pe)) - np.log(ds)) / (ff * z)
  
def gen_csv_data():
  A = [i for i in range(6, 101, 3)]
  F = [i for i in range(2, 35)]
  DS = [0.001, 0.01]
  EC = [0.001, 0.01]
  v = 1
  PE = [i/100 for i in range(1, 52, 5)]

  fname = "data/data_pod_beta.csv"
  print(fname)

  data = []

  for ds in DS:
    for ec in EC:
      for pe in PE:
        for aa in A:
          for ff in F:
            if ff > aa/3:
              continue

            beta = max_beta(aa, ff, v, ds, pe, ec)
            beta = min(beta, 1)
          
            # dl and du are "d * ds" as in the inequalities
            dl = lower(beta, ff, v, pe, ds)
            # dl = max(0.0, lower(beta, ff, v, pe, ds))
            du = upper(aa, ff, v, ds, pe, ec)

            data.append([aa, ff, pe, ds, ec, beta, dl, du])


  df = pd.DataFrame(data, columns = ['A', 'F', 'pe', 'ds', 'ec', 'beta', 'dl', 'du'])
  print(df)

  df.to_csv(fname, sep=',')

def get_bounds(aa, ff, v, ds, pe, ec):
  beta = max_beta(aa, ff, v, ds, pe, ec)

  dl = lower( min(1, beta), ff, v, pe, ds)
  du = upper(aa, ff, v, ds, pe, ec)
  
  print('A=%2d F=%2d ds=%.3f ec=%.3f pe=%.2f  |  beta=%.3f   dl = %.5f  du = %.5f' % (aa,ff,ds,ec,pe,beta,dl,du))
    
# -----------------------------------------------------
print('hello')

gen_csv_data()

# aa = 99
# v = 1
# for small in [0.001, 0.01]:
#   for pe in [0.16, 0.31, 0.52]:
#     for ff in [3, 9, 15, 21, 27, 33]:
#       ds = ec = small
#       get_bounds(aa, ff, v, ds, pe, ec)
