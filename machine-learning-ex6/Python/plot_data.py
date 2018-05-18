import matplotlib.pyplot as plt

def plot_data(X, y, ax, **params):
  ax.scatter(X[:, 0], X[:, 1], c=y, **params)
  plt.pause(max(0.2, len(X) / 400))