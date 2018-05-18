import matplotlib.pyplot as plt
import numpy as np

from plot_data import plot_data

def visualize_boundary(X, y, model):

  def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - h, x.max() + h
    y_min, y_max = y.min() - h, y.max() + h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy, x_min, x_max, y_min, y_max

  xx, yy, x_min, x_max, y_min, y_max = make_meshgrid(X[:, 0], X[:, 1])
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  fig, ax = plt.subplots()
  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)
  out = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
  plot_data(X, y, ax, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
  plt.pause(len(X) / 400)