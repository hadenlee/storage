import numpy as np
import pandas as pd
import seaborn as sns
import scipy.special as ss
import matplotlib.pyplot as plt

def find_opt_y_unlimited(v, d, x, e_c, d_s, p_e, a_i):
  z = -np.log(p_e)
  y = np.log(z) + np.log(x) + np.log(v+d) - np.log(d_s)
  y /= z
  return y

def find_opt_y(v, d, x, e_c, d_s, p_e, a_i):
  y = find_opt_y_unlimited(v, d, x, e_c, d_s, p_e, a_i)
  
  return max(1, min(x, int(y)))

def compute_prob(a_i, aa, x, ff):
  return ss.binom(a_i, x) * ss.binom(aa, ff-x) / ss.binom(aa+a_i, ff)

def compute_exp(v, d, e_c, d_s, p_e, a_i, aa, ff):
  exp = - e_c * a_i
  for x in range(1, min(a_i, ff)+1):
    q_x = compute_prob(a_i, aa, x, ff)
    opt_y = find_opt_y(v, d, x, e_c, d_s, p_e, a_i)
    prob_f = np.power(p_e, opt_y)
    exp += q_x * (v*x - prob_f * (d+v)*x - d_s * opt_y)
  return exp

def find_opt_a_i(v, d, e_c, d_s, p_e, t_i, aa, ff):
  max_exp, index = 0, 0
  for a_i in range(1, t_i+1):
    if a_i + aa < ff:
      continue
    exp = compute_exp(v, d, e_c, d_s, p_e, a_i, aa, ff)
    if exp > max_exp:
      max_exp, index = exp, a_i
  
  return max_exp, index

e_c, d_s = 0.000001, 0.000001
pe = [i/100. for i in range(1,11)]
v = 10
d = 200

F = [i for i in range(5, 16)]
T_i = [i for i in range(1, 21)]
A = [i for i in range(5, 51, 3)]
for ff in F:
  for t_i in T_i:
    for aa in A:
      if t_i + aa < ff:
        continue
      for p_e in pe:
        max_exp, index = find_opt_a_i(v, d, e_c, d_s, p_e, t_i, aa, ff)
        if index != t_i:
          print(v, d, e_c, d_s, p_e, t_i, aa, ff, max_exp, index)


#
#
#
#
# sns.set_theme(style="ticks")
#
# # Create a dataset with many short random walks
# rs = np.random.RandomState(4)
# pos = rs.randint(-1, 2, (20, 5)).cumsum(axis=1)
# pos -= pos[:, 0, np.newaxis]
# step = np.tile(range(5), 20)
# walk = np.repeat(range(20), 5)
# df = pd.DataFrame(np.c_[pos.flat, step, walk],
#                   columns=["position", "step", "walk"])
#
# # Initialize a grid of plots with an Axes for each walk
# grid = sns.FacetGrid(df, col="walk", hue="walk", palette="tab20c",
#                      col_wrap=4, height=1.5)
#
# # Draw a horizontal line to show the starting point
# grid.map(plt.axhline, y=0, ls=":", c=".5")
#
# # Draw a line plot to show the trajectory of each random walk
# grid.map(plt.plot, "step", "position", marker="o")
#
# # Adjust the tick positions and labels
# grid.set(xticks=np.arange(5), yticks=[-3, 3],
#          xlim=(-.5, 4.5), ylim=(-3.5, 3.5))
#
# # Adjust the arrangement of the plots
# grid.fig.tight_layout(w_pad=1)
#
# plt.show()
