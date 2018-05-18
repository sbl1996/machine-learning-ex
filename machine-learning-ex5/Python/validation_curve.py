import numpy as np

from train_linear_reg import train_linear_reg

def validation_curve(X, y, Xval, yval):
  m = len(X)

  lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

  error_train = np.zeros((len(lambda_vec),))
  error_val = np.zeros((len(lambda_vec),))

  for i in range(len(lambda_vec)):
    lamb = lambda_vec[i]
    theta = train_linear_reg(X, y, lamb)

    delta_t = X @ theta - y
    error_train[i] = (delta_t.T @ delta_t) / (2 * i)
    delta_cv = Xval @ theta - yval
    error_val[i] = (delta_cv.T @ delta_cv) / (2 * len(Xval))

  return lambda_vec, error_train, error_val