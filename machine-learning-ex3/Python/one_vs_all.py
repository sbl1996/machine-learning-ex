import numpy as np
from scipy.optimize import fmin_tnc

from lr_cost_function import lr_cost_function

def one_vs_all(X, y, num_labels, lamb):
  m, n = X.shape

  X = np.append(np.ones((m,1)), X, axis=1)
  
  all_theta = np.zeros((num_labels, n + 1))
  for c in range(1, num_labels + 1):
    initial_theta = np.zeros((n + 1,))
    eq = np.vectorize(lambda x: 1 if x == c else 0)(y)
    theta = fmin_tnc(lr_cost_function, initial_theta, args=(X, eq, lamb))[0]
    all_theta[c - 1, :] = theta

  return all_theta