import matplotlib.pyplot as plt

def plot_data(x, y):
  plt.plot(x, y, 'rx', markersize=10)
  plt.xlabel('Population of City in 10,000s')
  plt.ylabel('Profit in $10,000s')