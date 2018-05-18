import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
  # a1 = np.append(np.ones((X.shape[0], 1)), X, axis=1)
  # z2 = a1 @ Theta1.T
  # a2 = np.append(np.ones((z2.shape[0], 1)), sigmoid(z2), axis=1)
  # z3 = a2 @ Theta2.T
  # a3 = sigmoid(z3)
  m = len(X)
  a1 = np.append(np.ones((m, 1)), X, axis=1)
  a2 = np.append(np.ones((m, 1)), sigmoid(a1 @ Theta1.T), axis=1)
  a3 = sigmoid(a2 @ Theta2.T)
  return np.vectorize(lambda x: x + 1)(a3.argmax(axis=1))