import matplotlib.pyplot as plt
import numpy as np

def visualize_fit(X, mu, sigma2):
  xs = np.arange(0, 35.5, 0.5)
  X1, X2 = np.meshgrid(xs, xs)
  Z = multivariate_gaussian(np.c_[X1.ravel(), X2.ravel()], mu, sigma2)
  Z = Z.reshape(X1.shape)

  plt.plot(X[:, 0], X[:, 1], 'b.')
  plt.pause(0.1)
  plt.contour(X1, X2, Z, 10 ** np.arange(-20, 0, 3, 'float'))
  plt.pause(0.1)