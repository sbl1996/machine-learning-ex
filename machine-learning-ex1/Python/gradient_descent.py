import numpy as np

def gradient_descent(X, y, theta, alpha, num_iters):
  m = len(y)
  for _ in range(num_iters):
    theta = theta - (alpha / m) * X.T @ (X @ theta - y)
  return theta