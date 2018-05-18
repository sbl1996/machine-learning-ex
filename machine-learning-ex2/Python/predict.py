import numpy as np

from sigmoid import sigmoid

def predict(theta, X):
  h = sigmoid(X @ theta)
  return np.vectorize(lambda x: 1 if x >= 0.5 else 0)(h)