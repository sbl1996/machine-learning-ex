import numpy as np

def estimate_gaussian(X):
  m = len(X)
  mu = np.mean(X, axis=0)
  sigma2 = ((X - mu) ** 2 ).sum(axis=0) / m
  return mu, sigma2

