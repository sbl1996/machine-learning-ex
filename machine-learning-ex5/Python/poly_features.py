import numpy as np

def poly_features(X, p):
  X = X.ravel()
  return np.apply_along_axis(
    lambda i: X ** i, 0, np.arange(1, p + 1).reshape(1, p))