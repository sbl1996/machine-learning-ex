import numpy as np
import matplotlib.pyplot as plt

from poly_features import poly_features
from add_ones import add_ones

def plot_fit(min_x, max_x, mu, sigma, theta, p, ax):
  x = np.arange(min_x - 15, max_x + 25, 0.05)

  X_poly = poly_features(x, p)
  f = lambda i: (X_poly[:, i] - mu[i]) / sigma[i]
  X_poly = np.apply_along_axis(f, 0, np.arange(X_poly.shape[1]))
  X_poly = add_ones(X_poly)

  ax.plot(x, X_poly @ theta, '--', linewidth=2)
  plt.pause(0.1)