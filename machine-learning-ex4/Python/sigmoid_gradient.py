import numpy as np

from sigmoid import sigmoid

def sigmoid_gradient(z):
  f = sigmoid(z)
  g = f * (np.ones(f.shape) - f)
  return g