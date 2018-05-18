import numpy as np

def compute_numerical_gradient(J, theta):

  numgrad = np.zeros(theta.size)
  perturb = np.zeros(theta.size)

  e = 1e-4
  for p in range(0, theta.size):
    perturb[p] = e;
    loss1 = J(theta - perturb)[0]
    loss2 = J(theta + perturb)[0]
    numgrad[p] = (loss2 - loss1) / (2*e)
    perturb[p] = 0

  return numgrad