from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from tabulate import tabulate

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

def plots3(df, prefix, ds_val, col_name):
  print('obtaining a subset of data')
  ds = df.copy(deep=True)  
  
  ds = (ds[(ds['ds'] == ds_val)])
  ds = (ds[ (ds['pe']>= 0.01) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  ds = (ds[(ds['F'] == ds['A']/3)])
  
  ds = (ds[ 
    (ds['beta']== 0.10) | (ds['beta']== 0.20) | (ds['beta']== 0.30) | (ds['beta']== 0.40) | (ds['beta']== 0.50)
  | (ds['beta']== 0.60) | (ds['beta']== 0.70) | (ds['beta']== 0.80) | (ds['beta']== 0.90) | (ds['beta']== 1.00) 
  ])

  print(ds)

  cols = ["A", "beta", "pe", col_name]
  ds = ds[cols]

  print(ds.info())
  print(tabulate(ds, headers='keys', tablefmt='psql'))

  title = '%s.png' % (prefix)
  save_plot(ds, title, "pe", col_name, "beta", "A", 4)



def plots2(df, prefix, ds_val):
  print('obtaining a subset of data')
  ds = df.copy(deep=True)  
  
  ds = (ds[(ds['ds'] == ds_val)])
  ds = (ds[ (ds['pe']>= 0.01) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  ds = (ds[(ds['F'] == ds['A']/3)])
  ds = (ds[(ds['max_ec'] >= 0)])
  ds = (ds[ 
    (ds['beta']== 0.10) | (ds['beta']== 0.20) | (ds['beta']== 0.30) | (ds['beta']== 0.40) | (ds['beta']== 0.50)
  | (ds['beta']== 0.60) | (ds['beta']== 0.70) | (ds['beta']== 0.80) | (ds['beta']== 0.90) | (ds['beta']== 1.00) 
  ])

  print(ds)

  cols = ["A", "beta", "pe", "max_ec"]
  ds = ds[cols]

  print(ds.info())
  print(tabulate(ds, headers='keys', tablefmt='psql'))

  title = '%s.png' % (prefix)
  save_plot(ds, title, "pe", "max_ec", "beta", "A", 4)


def plots1(df, prefix, ds_val):
  print('obtaining a subset of data')
  ds = df.copy(deep=True)  
  
  ds = (ds[(ds['ds'] == ds_val)])
  ds = (ds[ (ds['pe']>= 0.01) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  ds = (ds[(ds['max_ec'] >= 0)])
  ds = (ds[(ds['F'] == ds['A']/3)])
  ds = (ds[ 
    (ds['beta']== 0.10) | (ds['beta']== 0.20) | (ds['beta']== 0.30) | (ds['beta']== 0.40) | (ds['beta']== 0.50)
  | (ds['beta']== 0.60) | (ds['beta']== 0.70) | (ds['beta']== 0.80) | (ds['beta']== 0.90) | (ds['beta']== 1.00) 
  ])

  print(ds)

  cols = ["A", "beta", "pe", "min_d"]
  ds = ds[cols]

  print(ds.info())
  print(tabulate(ds, headers='keys', tablefmt='psql'))

  title = '%s.png' % (prefix)
  save_plot(ds, title, "pe", "min_d", "beta", "A", 4)


def plot_3d(df, valA):
  ds = df.copy(deep=True)
  ds = (ds[(ds['ds'] == 0.001)])
  ds = (ds[(ds['ec'] == 0.001)])
  ds = (ds[ (ds['pe']>= 0.01) & (ds['pe'] <= 0.41) ])
  ds = (ds[(ds['A'] == valA)])
  ds = (ds[(ds['F'] == valA//3)])
  ds = (ds[(ds['max_d'] >= ds['min_d'])])
  # ds = (ds[(ds['max_d'] >= 0)])
  
  # ds = (ds[(ds['A'] % 9 == 0) & (ds['A'] > 30)])
  # ds = (ds[(ds['F'] == ds['A']/3)])

  # ds = (ds[
  #   (ds['beta']== 0.11) | (ds['beta']== 0.21) | (ds['beta']== 0.31) | (ds['beta']== 0.41) | (ds['beta']== 0.51)
  # # | (ds['beta']== 0.60) | (ds['beta']== 0.70) | (ds['beta']== 0.80) | (ds['beta']== 0.90) | (ds['beta']== 1.00)
  # ])

  print(ds)
  

  cols = ["pe", "beta", "max_d"]

  # And transform the old column name in something numeric
  # ds['X']=pd.Categorical(ds['X'])
 #  ds['X']=ds['X'].cat.codes
 #
  # Make the plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_trisurf(ds['pe'], ds['beta'], ds['min_d'], cmap=plt.cm.viridis, linewidth=0.2)
  plt.show()
 
  # # to Add a color bar which maps values to colors.
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # surf=ax.plot_trisurf(ds['Y'], ds['X'], ds['Z'], cmap=plt.cm.viridis, linewidth=0.2)
  # fig.colorbar( surf, shrink=0.5, aspect=5)
  # plt.show()
  #
  # # Rotate it
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # surf=ax.plot_trisurf(ds['Y'], ds['X'], ds['Z'], cmap=plt.cm.viridis, linewidth=0.2)
  # ax.view_init(30, 45)
  # plt.show()
  #
  # # Other palette
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # ax.plot_trisurf(ds['Y'], ds['X'], ds['Z'], cmap=plt.cm.jet, linewidth=0.01)
  # plt.show()

# -----------------------------------------------------
print('hello')

df = pd.read_csv('data/data_range_on_d.csv', usecols=range(1,9))

# df.apply(pd.to_numeric)

print(df)
print(df.info())

plot_3d(df,30)
plot_3d(df,60)
plot_3d(df,90)


# plots1(df, "robust-d-when-ds-1000th", 0.001)
# plots1(df, "robust-d-when-ds-100th", 0.01)

# plots2(df, "robust-ec-when-ds-1000th", 0.001)
# plots2(df, "robust-ec-when-ds-100th", 0.01)


# plots3(df, "robust-ec1-when-ds-1000th", 0.001, 'max_ec1')
# plots3(df, "robust-ec1-when-ds-100th", 0.01, 'max_ec1')
# plots3(df, "robust-ec2-when-ds-1000th", 0.001, 'max_ec2')
# plots3(df, "robust-ec2-when-ds-100th", 0.01, 'max_ec2')
# plots3(df, "robust-ec3-when-ds-1000th", 0.001, 'max_ec3')
# plots3(df, "robust-ec3-when-ds-100th", 0.01, 'max_ec3')


