import numpy as np
from compute_cost import compute_cost

def gradient_descent_h(X, y, theta, alpha, num_iters):
  m = len(y)
  J_history = np.zeros(num_iters)
  for i in range(num_iters):
    theta = theta - (alpha / m) * X.T @ (X @ theta - y)
    J_history[i] = compute_cost(X, y, theta)
  return theta, J_history