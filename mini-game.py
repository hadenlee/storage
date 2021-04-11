import numpy as np
import pandas as pd
import scipy.special as ss

def find_opt_y_unlimited(v, d, x, d_s, p_e):
  z = -np.log(p_e)
  y = np.log(z) + np.log(x) + np.log(v + d * d_s) - np.log(d_s)
  y /= z
  return y

def find_opt_y(v, d, x, d_s, p_e):
  y = find_opt_y_unlimited(v, d, x, d_s, p_e)
  
  return max(1, min(x, int(y)))

def compute_prob(a_i, aa, x, ff):
  return ss.binom(a_i, x) * ss.binom(aa, ff-x) / ss.binom(aa+a_i, ff)

def compute_fx(v, x, prob_f, d, d_s, opt_y):
  return v*x - prob_f * (d * d_s + v) * x - d_s * opt_y

def compute_exp_rev(v, e_c, a_i, aa, ff, fx_vals):
  exp = - e_c * a_i

  for x in range(1, min(a_i, ff)+1):
    q_x = compute_prob(a_i, aa, x, ff)    
    fx = fx_vals[x]
    exp += q_x * fx 

  return exp

def compute_fx_all(v, d, d_s, p_e, t_i):
  fx_vals = [0]
  for x in range(1, t_i+1):
    opt_y = find_opt_y(v, d, x, d_s, p_e)
    prob_f = np.power(p_e, opt_y)
    fx = compute_fx(v, x, prob_f, d, d_s, opt_y)
    fx_vals.append(fx)

  # sanity check: f(x) - f(x-1) >= f(1) must be true for all x. check this.
  for x in range(1, t_i+1):
    diff = fx_vals[x] - fx_vals[x-1]
    if diff < fx_vals[1]:
      print('Weird: ', fx_vals)
      raise Exception('sanity check on f(x) failed.')

  return fx_vals

  
def find_opt_a_i(v, d, e_c, t_i, aa, ff, fx_vals):
  max_exp, index = 0, 0

  for a_i in range(1, t_i+1):
    if a_i + aa < ff:
      continue
    exp = compute_exp_rev(v, e_c, a_i, aa, ff, fx_vals)
    if exp > max_exp:
      max_exp, index = exp, a_i
  
  return max_exp, index

# -----------------------------------------------------------
e_c, d_s = 0.001, 0.001
v = 1

pe = [i/100. for i in range(1, 42, 5)]
F = [i for i in range(5, 21)]
T_i = [i for i in range(1, 21)]
A = [i for i in range(5, 51)]
D = [1, 10, 100, 1000, 10000]

fname = "data_F%d_T%d_A%d_D%d_pe%d.csv" % (max(F), max(T_i), max(A), max(D), 100*max(pe))
print(fname)

data = []

for d in D:
  for p_e in pe:
    print('d = {} p_e = {}'.format(d, p_e))
    for t_i in T_i:
      fx_vals = compute_fx_all(v, d, d_s, p_e, t_i)
      for ff in F:
        for aa in A:
          if t_i + aa < ff:
            continue
          max_exp, index = find_opt_a_i(v, d, e_c, t_i, aa, ff, fx_vals)
          if index == 0:
            print(v, d, e_c, d_s, p_e, t_i, aa, ff, max_exp, index)
            print('skip: because not participating is better.')
            continue
          if index != t_i:
            print(v, d, e_c, d_s, p_e, t_i, aa, ff, max_exp, index)
            raise Exception('conjecture is incorrect')
          data.append([d, ff, t_i, aa, p_e, index, max_exp])

df = pd.DataFrame(data, columns = ['d', 'F', 'T_i', 'A', 'p_e', 'a_opt', 'rev'])
print(df)

df.to_csv(fname, sep=',')

