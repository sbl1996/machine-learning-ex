import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
  m = len(X)
  num_labels = Theta2.shape[0]

  p = np.zeros((X.shape[0]),)

  h1 = sigmoid(np.append(np.ones((m, 1)), X, axis=1) @ Theta1.T)
  h2 = sigmoid(np.append(np.ones((m, 1)), h1, axis=1) @ Theta2.T)

  return h2.argmax(axis=1) + 1