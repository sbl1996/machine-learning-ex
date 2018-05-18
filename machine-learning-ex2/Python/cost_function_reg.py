import numpy as np
from sigmoid import sigmoid

def cost_function_reg(theta, X, y, lam):
  m = len(y)
  h = sigmoid(X @ theta)
  J = - (y @ np.log(h) \
      + (np.ones(y.shape) - y) @ np.log(np.ones(h.shape) - h)) / m \
      + lam / (2 * m) * (theta @ theta)
  L = np.ones(theta.shape)
  L[0] = 0
  grad = (X.T @ (h - y)) / m + lam / m * L * theta
  return J, grad