import numpy as np

def normalize_ratings(Y, R):
  m, n = Y.shape
  Ymean = np.zeros((m,))
  Ynorm = np.zeros(Y.shape)

  for i in range(m):
    idx = R[i] == 1
    Ymean[i] = Y[i, idx].mean()
    Ynorm[i, idx] = Y[i, idx] - Ymean[i]

  return Ynorm, Ymean