import numpy as np

from train_linear_reg import train_linear_reg

def learning_curve(X, y, Xval, yval, lamb):
  m = len(X)

  error_train = np.zeros((m, 1))
  error_val = np.zeros((m,1))

  for i in range(m):
    X_ = X[:i+1, :]
    y_ = y[:i+1]
    theta = train_linear_reg(X_, y_, lamb)
    delta_t = X_ @ theta - y_
    error_train[i] = (delta_t.T @ delta_t) / (2 * m)
    delta_cv = Xval @ theta - yval
    error_val[i] = (delta_cv.T @ delta_cv) / (2 * len(Xval))

  return error_train, error_val