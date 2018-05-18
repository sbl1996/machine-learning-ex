import numpy as np

def linear_reg_cost_function(X, y, theta, lamb):
  m = len(X) 
  theta_ = theta[1:]
  delta = X @ theta - y
  J = delta.T @ delta / (2 * m) + (lamb / (2 * m)) * (theta_.T @ theta_)
  grad = (X.T @ delta) / m + (lamb / m) * np.insert(theta_, 0, 0)

  return J, grad