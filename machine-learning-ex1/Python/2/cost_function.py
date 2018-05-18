def cost_function(theta, X, y):
  m = len(X)
  z = X @ theta - y
  J = (z @ z) / (2 * m)
  grad = X.T @ z / m
  return J, grad