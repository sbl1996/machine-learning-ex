import numpy as np
from sigmoid import sigmoid

def lr_cost_function(theta, X, y, lamb):
  m = len(y)
  z = X @ theta
  h = sigmoid(z)
  theta1 = theta.copy()
  theta1[0] = 0
  J = (y @ np.log(h) + \
      (np.ones(y.shape) - y) @ np.log(np.ones(z.shape) - h)) / (-m) + \
      (lamb / (2 * m)) * (theta1 @ theta1)
  grad = X.T @ (sigmoid(z) - y) / m + (lamb / m) * theta1
  return J, grad