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



# -----------------------------------------------------
print('hello')

df = pd.read_csv('data/data_F20_T20_A50_D81_pe31.csv', usecols=range(1,8))
df.apply(pd.to_numeric)

print(df)
print(df.info())

# plot 1
# plots_Ti_1(df, 'charts/plot1_By_Ti')

# plot 2
#plots_Ti_2(df, 'charts/plot2_By_Ti')


