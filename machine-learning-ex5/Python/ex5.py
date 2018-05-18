## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

from linear_reg_cost_function import linear_reg_cost_function
from train_linear_reg import train_linear_reg
from learning_curve import learning_curve
from poly_features import poly_features
from feature_normalize import feature_normalize
from add_ones import add_ones
from validation_curve import validation_curve
from plot_fit import plot_fit
## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

plt.ion()

# Load Training Data
print('Loading and Visualizing Data ...\n')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = loadmat('ex5data1.mat')
X = data['X']
Xtest = data['Xtest']
Xval = data['Xval']
y = data['y'].ravel()
ytest = data['ytest'].ravel()
yval = data['yval'].ravel()


# m = Number of examples
m = len(X)

# Plot training data
fig, ax = plt.subplots()
ax.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.pause(0.1)
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')

print('Program paused. Press enter to continue.\n')
input()

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([1, 1])
J, grad = linear_reg_cost_function(add_ones(X), y, theta, 1)

print('Cost at theta = [1 ; 1]: %s \
\n(this value should be about 303.993192)\n' % J)

print('Program paused. Press enter to continue.\n')
input()

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([1, 1])
J, grad = linear_reg_cost_function(add_ones(X), y, theta, 1)

print('Gradient at theta = [1 ; 1]:  [%s; %s] \
\n(this value should be about [-15.303016; 598.250744])\n' % (grad[0], grad[1]))

print('Program paused. Press enter to continue.\n')
input()


## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
lamb = 0
theta = train_linear_reg(add_ones(X), y, lamb)

# Plot fit over the data
fig, ax = plt.subplots()
ax.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.pause(0.1)
ax.set_xlabel('Change in water level (x)')
ax.set_ylabel('Water flowing out of the dam (y)')
ax.plot(X, add_ones(X) @ theta, '--', linewidth=2)
plt.pause(0.1)

print('Program paused. Press enter to continue.\n')
input()

## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

lamb = 0
error_train, error_val = \
    learning_curve(add_ones(X), y, \
                   add_ones(Xval), yval, \
                   lamb)

fig, ax = plt.subplots()
ax.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val)
plt.pause(0.1)
ax.set_title('Learning curve for linear regression')
ax.legend(['Train', 'Cross Validation'])
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Error')
ax.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
  print('  \t%d\t\t%f\t%f\n' % (i, error_train[i], error_val[i]))

print('Program paused. Press enter to continue.\n')
input()

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = poly_features(X, p)
X_poly, mu, sigma = feature_normalize(X_poly)      # Normalize
X_poly = add_ones(X_poly) # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = poly_features(Xtest, p)
f = lambda i: (X_poly_test[:, i] - mu[i]) / sigma[i]
X_poly_test = np.apply_along_axis(f, 0, np.arange(X_poly_test.shape[1]))
X_poly_test = add_ones(X_poly_test) # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = poly_features(Xval, p)
f = lambda i: (X_poly_val[:, i] - mu[i]) / sigma[i]
X_poly_val = np.apply_along_axis(f, 0, np.arange(X_poly_val.shape[1]))
X_poly_val = add_ones(X_poly_val) # Add Ones

print('Normalized Training Example 1:\n')
print('  %s  \n' % X_poly[0, :])

print('\nProgram paused. Press enter to continue.\n')
input()



## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lamb = 0
theta = train_linear_reg(X_poly, y, lamb)

# Plot training data and fit
fig1, ax1 = plt.subplots()
ax1.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.pause(0.1)
plot_fit(X.min(), X.max(), mu, sigma, theta, p, ax1)
ax1.set_xlabel('Change in water level (x)')
ax1.set_ylabel('Water flowing out of the dam (y)')
ax1.set_title('Polynomial Regression Fit (lambda = %s)' % lamb)

fig2, ax2 = plt.subplots()
error_train, error_val = \
    learning_curve(X_poly, y, X_poly_val, yval, lamb)
ax2.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val)
plt.pause(0.1)
ax2.set_title('Polynomial Regression Learning Curve (lambda = %s)' % lamb)
ax2.set_xlabel('Number of training examples')
ax2.set_ylabel('Error')
ax2.axis([0, 13, 0, 100])
ax2.legend(['Train', 'Cross Validation'])

print('Polynomial Regression (lambda = %s)\n\n' % lamb)
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
  print('  \t%d\t\t%f\t%f\n' % (i, error_train[i], error_val[i]))

print('Program paused. Press enter to continue.\n')
input()

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = \
  validation_curve(X_poly, y, X_poly_val, yval)

fig, ax = plt.subplots()
ax.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.pause(0.1)
ax.legend(['Train', 'Cross Validation'])
ax.set_xlabel('lambda')
ax.set_ylabel('Error')

print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(len(lambda_vec)):
  print(' %f\t%f\t%f\n' % (lambda_vec[i], error_train[i], error_val[i]))

print('Program paused. Press enter to continue.\n')
input()