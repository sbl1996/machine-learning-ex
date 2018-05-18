import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from compute_cost import compute_cost
from plot_data import plot_data
from gradient_descent import gradient_descent

def ex1():
  print('Plotting Data ...\n')
  data = pd.read_csv('ex1data1.txt', header=None).as_matrix()
  rX = data[:, [0]]
  y = data[:, [1]]
  m = len(rX)

  plot_data(rX, y)
    
  input('Program paused. Press enter to continue.\n')

  X = np.append(np.ones((m,1)), rX, 1)

  theta = np.zeros((2,1))

  iterations = 1500
  alpha = 0.01

  print('\nTesting the cost function ...\n')
  J = compute_cost(X, y, theta)
  print('With theta = [0 ; 0]\nCost computed = %f\n' % (J))
  print('Expected cost value (approx) 32.07\n');

  J = compute_cost(X, y, np.array([[-1], [2]]))
  print('With theta = [-1 ; 2]\nCost computed = %f\n' % (J))
  print('Expected cost value (approx) 54.24\n');

  input('Program paused. Press enter to continue.\n')


  print('\nRunning Gradient Descent ...\n')
  theta = gradient_descent(X, y, theta, alpha, iterations)

  print('Theta found by gradient descent:\n')
  print('%s\n' % theta)
  print('Expected theta values (approx)\n')
  print(' -3.6303\n  1.1664\n\n')

  plt.plot(rX, y, 'rx')
  plt.plot(rX, X.dot(theta), '-')
  plt.legend(['Training data', 'Linear regression'])

  predict1 = np.array([[1, 3.5]]).dot(theta)
  print('For population = 35,000, we predict a profit of %f\n' % (predict1 * 10000))

  predict2 = np.array([[1, 7]]).dot(theta)
  print('For population = 70,000, we predict a profit of %f\n' % (predict2 * 10000))

  input('Program paused. Press enter to continue.\n')


  print('Visualizing J(theta_0, theta_1) ...\n')

  theta0_vals = np.linspace(-10, 10, 100)
  theta1_vals = np.linspace(-1, 4, 100)

  J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

  for i in range(len(theta0_vals)):
      for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = compute_cost(X, y, t)

  J_vals0 = J_vals.T
  theta0_vals0, theta1_vals0 = np.meshgrid(theta0_vals, theta1_vals)

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(theta0_vals0, theta1_vals0, J_vals0, cmap=cm.coolwarm)
  ax.invert_xaxis()
  ax.set_xlabel(r'$\theta_{0}$')
  ax.set_ylabel(r'$\theta_{1}$')
  ax.set_zlabel(r'$J(\theta_{0},\theta_{1})$')

  input()

if __name__ == '__main__':
  ex1()