import numpy as np
from sigmoid import sigmoid

def cost_function(theta, X, y):
  m = len(y)
  h = sigmoid(X @ theta)
  J = - (y @ np.log(h) \
      + (np.ones(y.shape) - y) @ np.log(np.ones(h.shape) - h)) / m
  grad = (X.T @ (h - y)) / m
  return J, grad