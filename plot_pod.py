import numpy as np
import pandas as pd
import scipy.special as ss

import seaborn as sns
import matplotlib.pyplot as plt

def save_plot(d_sub, title, col_x, col_y, col_hue, col_col, col_wrap):
  # Colors within a single chart: col_hue (3rd arg)
  sns.set_style("whitegrid")

  g = sns.catplot(x=col_x, y=col_y, hue=col_hue, col=col_col,
                  capsize=.2, palette="tab10", 
                  height=6, aspect=.75,
                  col_wrap=col_wrap,
                  kind="point", data=d_sub)
  g.despine(left=True)
  g.savefig(title)


def plots1(df, prefix, ds_val, ec_val):
  print('obtaining a subset of data')
  ds = df.copy(deep=True)  
  
  ds = (ds[ (ds['pe']>= 0.01) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['ds'] == ds_val) & (ds['ec'] == ec_val)])
  ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  ds = (ds[(ds['F'] % 6 == 3)])

  print(ds)

  cols = ["A", "F", "pe", "beta"]
  ds = ds[cols]

  print(ds.info())

  title = '%s.png' % (prefix)
  save_plot(ds, title, "pe", "beta", "F", "A", 4)


def plots2a(df, prefix, ds_val, ec_val):
  print('obtaining a subset of data')
  ds = df.copy(deep=True)  
  
  ds = (ds[ (ds['pe']>= 0.01) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['ds'] == ds_val) & (ds['ec'] == ec_val)])
  ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  ds = (ds[(ds['F'] % 6 == 3)])

  print(ds)

  cols = ["A", "F", "pe", "dl"]
  ds = ds[cols]

  print(ds.info())

  title = '%s.png' % (prefix)
  save_plot(ds, title, "pe", "dl", "F", "A", 4)


def plots2b(df, prefix, ds_val, ec_val, pe_l=0.01):
  print('obtaining a subset of data')
  ds = df.copy(deep=True)  
  
  ds = (ds[ (ds['pe']>= pe_l) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['ds'] == ds_val) & (ds['ec'] == ec_val)])
  ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  ds = (ds[(ds['F'] % 6 == 3)])

  print(ds)

  cols = ["A", "F", "pe", "du"]
  ds = ds[cols]

  print(ds.info())

  title = '%s.png' % (prefix)
  save_plot(ds, title, "pe", "du", "F", "A", 4)


# -----------------------------------------------------
print('hello')

df = pd.read_csv('data/data_pod_beta.csv', usecols=range(1,9))

# df.apply(pd.to_numeric)

print(df)
print(df.info())


# plots1(df, "pod-max-beta-ds-1000th-ec-1000th", 0.001, 0.001)
# plots1(df, "pod-max-beta-ds-1000th-ec-100th", 0.001, 0.01)
#
# plots1(df, "pod-max-beta-ds-100th-ec-1000th", 0.01, 0.001)
# plots1(df, "pod-max-beta-ds-100th-ec-100th", 0.01, 0.01)

plots2a(df, "pod-penalty-ds-1000th-ec-1000th", 0.001, 0.001)
plots2a(df, "pod-penalty-ds-1000th-ec-100th", 0.001, 0.01)
plots2a(df, "pod-penalty-ds-100th-ec-1000th", 0.01, 0.001)
plots2a(df, "pod-penalty-ds-100th-ec-100th", 0.01, 0.01)

# plots2b(df, "pod-IR-ds-1000th-ec-1000th", 0.001, 0.001)
# plots2b(df, "pod-IR-ds-1000th-ec-100th", 0.001, 0.01)
# plots2b(df, "pod-IR-ds-100th-ec-1000th", 0.01, 0.001)
# plots2b(df, "pod-IR-ds-100th-ec-100th", 0.01, 0.01)

# plots2b(df, "pod-IRc-ds-1000th-ec-1000th", 0.001, 0.001, 0.21)
# plots2b(df, "pod-IRc-ds-1000th-ec-100th", 0.001, 0.01, 0.21)
# plots2b(df, "pod-IRc-ds-100th-ec-1000th", 0.01, 0.001, 0.21)
# plots2b(df, "pod-IRc-ds-100th-ec-100th", 0.01, 0.01, 0.21)
