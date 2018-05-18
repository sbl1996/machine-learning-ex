import numpy as np
import pandas as pd

def compute_cost(X, y, theta):
  m = len(y)
  diff = X @ theta - y
  J = diff.T @ diff / (2 * m)
  return J[0, 0]