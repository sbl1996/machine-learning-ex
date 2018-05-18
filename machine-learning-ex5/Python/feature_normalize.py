import numpy as np

def feature_normalize(X):
  X_norm = np.zeros(X.shape)
  mu = np.apply_along_axis(lambda v: v.mean(), 0, X)
  sigma = np.apply_along_axis(lambda v: v.std(), 0, X)
  for i in range(X_norm.shape[1]):
    X_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

  return X_norm, mu, sigma