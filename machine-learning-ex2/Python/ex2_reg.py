## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from cost_function_reg import cost_function_reg
from cost_function import cost_function
from predict import predict
from plot_data import plot_data
from plot_decision_boundary import plot_decision_boundary
from sigmoid import sigmoid
from map_feature import map_feature

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

plot_data(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

# Specified in plot order
plt.legend(['y = 1', 'y = 0'])


## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#
 
# Add Polynomial Features
 
# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = map_feature(X[:,0], X[:,1])
 
# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1]),)
 
# Set regularization parameter lambda to 1
lam = 1
 
# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = cost_function_reg(initial_theta, X, y, lam)

print('Cost at initial theta (zeros): %s\n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(' %s \n' % grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

print('\nProgram paused. Press enter to continue.\n')
input()

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1],))
cost, grad = cost_function_reg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): %s\n' % cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(' %s \n' % grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

print('\nProgram paused. Press enter to continue.\n')
input()

## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1],)

# Set regularization parameter lambda to 1 (you should vary this)

# theta, J, _ = fmin_tnc(cost_function, initial_theta, \
#   None, (X, y), maxfun=100*len(X))

# # Plot Boundary
# plot_decision_boundary(theta, X, y)

# Optimize
lam = 1
theta, J, _ = fmin_tnc(cost_function_reg, initial_theta, \
  None, (X, y, lam))

# Plot Boundary
plot_decision_boundary(theta, X, y)
plt.title('lambda = %g' % lam)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %s\n' % (np.double(p == y).mean() * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')

