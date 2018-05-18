import numpy as np

def compute_numerical_gradient(f, theta):
  numgrad = np.zeros(theta.shape)
  perturb = np.zeros(theta.shape)
  e = 1e-4
  for p in range(theta.size):
    perturb[p] = e
    loss1 = f(theta - perturb)[0]
    loss2 = f(theta + perturb)[0]
    numgrad[p] = (loss2 - loss1) / (2 * e)
    perturb[p] = 0

  return numgrad