import numpy as np

def add_ones(X):
  return np.append(np.ones((X.shape[0],1)), X, axis=1)