import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from compute_cost import compute_cost
from gradient_descent import gradient_descent
from plot_data import plot_data
# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

# ==================== Part 1: Basic Function ====================
# Complete warmup_exercise.py
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmup_exercise())

print('Program paused. Press enter to continue.\n')
input()


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, :1]
y = data[:, [-1]]
m = len(y) # number of training examples

# Plot Data
# Note: You have to complete the code in plot_data.py
plot_data(X, y)

print('Program paused. Press enter to continue.\n')
input()


# =================== Part 3: Cost and Gradient descent ===================

X = np.append(np.ones((m, 1)), X, axis=1) # Add a column of ones to x
theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = compute_cost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = %f\n'% J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = compute_cost(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 54.24\n')

print('Program paused. Press enter to continue.\n')
input()

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta = gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('%s\n' % theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(X[:, [1]], X @ theta, '-')
plt.legend(['Training data', 'Linear regression'])

# Predict values for population sizes of 35,000 and 70,000
predict1 = (np.array([[1, 3.5]]) @ theta)[0, 0]
print('For population = 35,000, we predict a profit of %f\n' % (predict1 * 10000))
predict2 = (np.array([[1, 7]]) @ theta)[0, 0]
print('For population = 70,000, we predict a profit of %f\n' % (predict2 * 10000))

print('Program paused. Press enter to continue.\n')
input()

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
      t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
      J_vals[i,j] = compute_cost(X, y, t)


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
# Surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm)
ax.set_xlabel(r'$\theta_{0}$')
ax.set_ylabel(r'$\theta_{1}$')
ax.set_zlabel(r'$J(\theta_{0},\theta_{1})$')

# Contour plot
figure, ax = plt.subplots()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
ax.set_xlabel(r'$\theta_{0}$')
ax.set_ylabel(r'$\theta_{1}$')
ax.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
