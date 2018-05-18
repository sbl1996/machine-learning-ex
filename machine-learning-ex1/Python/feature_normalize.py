import numpy as np

def feature_normalize(X):
  def f(v):
    return (v - v.mean()) / v.std()
  return np.apply_along_axis(f, 0, X)