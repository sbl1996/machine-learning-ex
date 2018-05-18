import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin_tnc
from gradient_descent import gradient_descent
from cost_function import cost_function
from feature_normalize import feature_normalize

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
m = len(X)

means = np.apply_along_axis(lambda v: v.mean(), 0, X)
stds = np.apply_along_axis(lambda v: v.std(), 0, X)
f = lambda v: np.insert((v - means) / stds, 0, 1)
X = np.apply_along_axis(f, 1, X)
initial_theta = np.zeros((X.shape[1],))

theta = gradient_descent(cost_function, initial_theta, (X,y))
theta[0] = theta[0] - (theta[1:] @ (means / stds))
theta = theta[1:] / stds

# theta = fmin_tnc(cost_function, initial_theta, args=(X,y))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], y)
ax.plot_trisurf(X[:, 1], X[:, 2], X @ theta)

price = np.array([1650, 3]) @ theta