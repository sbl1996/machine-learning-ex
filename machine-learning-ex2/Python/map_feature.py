import numpy as np

def map_feature(X1, X2, degree=6):
  n = (degree + 3) * degree // 2 + 1
  out = np.ones(n)
  if isinstance(X1, np.ndarray):
    out = np.ones((len(X1), n))
    cnt = 1
    for i in range(1, degree + 1):
      for j in range(i + 1):
        out[:, cnt] = (X1 ** (i - j)) * (X2 ** j)
        cnt += 1
  else:
    cnt = 1
    for i in range(1, degree + 1):
      for j in range(i + 1):
        out[cnt] = (X1 ** (i - j)) * (X2 ** j)
        cnt += 1
  return out