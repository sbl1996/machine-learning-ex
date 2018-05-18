import numpy as np
from scipy import linalg

def gaussian_kernel(x1, x2, sigma):
  return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))