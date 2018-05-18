import numpy as np
from scipy.optimize import fmin_tnc

from linear_reg_cost_function import linear_reg_cost_function

def train_linear_reg(X, y, lamb):
  initial_theta = np.zeros((X.shape[1],))
  cost_func = lambda t: linear_reg_cost_function(X, y, t, lamb)
  theta = fmin_tnc(cost_func, initial_theta)[0]
  return theta 