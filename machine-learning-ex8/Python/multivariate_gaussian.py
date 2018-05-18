import numpy as np

def multivariate_gaussian(X, mu, Sigma2):
  k = len(mu);
  if Sigma2.ndim == 1:
    Sigma2 = np.diag(Sigma2)

  X = X - mu
  p = (2 * np.pi) ** (-k / 2) * linalg.det(Sigma2) ** (-0.5) * \
      np.exp(-0.5 * (X @ linalg.pinv(Sigma2) * X).sum(axis=1))

  return p 