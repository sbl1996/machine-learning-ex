import numpy as np
from sigmoid import sigmoid

def predict_one_vs_all(all_theta, X):
  m = len(X)
  num_labels = all_theta.shape[0]

  X = np.append(np.ones((m, 1)), X, axis=1)
  z = sigmoid(X @ all_theta.T)
  return np.vectorize(lambda x: x + 1)(z.argmax(axis=1))