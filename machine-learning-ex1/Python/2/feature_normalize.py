import numpy as np

def feature_normalize(X):
  means = np.apply_along_axis(lambda v: v.mean(), 0, X)
  stds = np.apply_along_axis(lambda v: v.std(), 0, X)
  def f(v):
    return (v - means) / stds
  return f