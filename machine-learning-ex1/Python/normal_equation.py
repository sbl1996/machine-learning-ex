import numpy as np
from scipy import linalg

def normal_equation(X, y):
  theta = linalg.inv(X.T @ X) @ X.T @ y
  return theta