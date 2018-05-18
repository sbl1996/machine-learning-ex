import matplotlib.pyplot as plt
import numpy as np
from plot_data import plot_data
from map_feature import map_feature

def plot_decision_boundary(theta, X, y):
  #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
  #the decision boundary defined by theta
  #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
  #   positive examples and o for the negative examples. X is assumed to be 
  #   a either 
  #   1) Mx3 matrix, where the first column is an all-ones column for the 
  #      intercept.
  #   2) MxN, N>3 matrix, where the first column is all-ones

  # Plot Data
  plot_data(X[:, 1:3], y)

  if X.shape[1] <= 3:
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = np.array([X[:,1].min() - 2, X[:,1].max() + 2])

    # Calculate the decision boundary line
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

    # Plot, and adjust axes for better viewing
    plt.plot(plot_x, plot_y)
    
    # Legend, specific for the exercise
    plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
  else:
    # Here is the grid range
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    # Evaluate z = theta*x over the grid
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = map_feature(u[i], v[j]) @ theta
    z = z.T # important to transpose z before calling contour
    plt.contour(u, v, z, [0], linewidths=2)
    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    