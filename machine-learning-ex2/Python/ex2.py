## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from cost_function import cost_function
from predict import predict
from plot_data import plot_data
from plot_decision_boundary import plot_decision_boundary
from sigmoid import sigmoid
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o \
indicating (y = 0) examples.\n')

plot_data(X, y)

# Put some labels 
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend(['Admitted', 'Not admitted'])

print('\nProgram paused. Press enter to continue.\n')
input()


## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.append(np.ones((m, 1)), X, axis=1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1,))

# Compute and display initial cost and gradient
cost, grad = cost_function(initial_theta, X, y)

print('Cost at initial theta (zeros): %s\n' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n')
print(' %s \n' % grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = cost_function(test_theta, X, y)

print('\nCost at test theta: %s\n' % cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(' %s \n' % grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

print('\nProgram paused. Press enter to continue.\n')
input()


## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
theta, cost, _ = fmin_tnc(cost_function, initial_theta, \
  None, (X, y))

# Print theta to screen
print('Cost at theta found by fminunc: %s\n' % cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n')
print(' %s \n' % theta)
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plot_decision_boundary(theta, X, y)

# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

# Specified in plot order
plt.legend('Admitted', 'Not admitted')

print('\nProgram paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.array([1, 45, 85]) @ theta)
print('For a student with scores 45 and 85, we predict an admission\n\
   probability of %s\n' % prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %s\n' % (np.double(p == y).mean() * 100))
print('Expected accuracy (approx): 89.0\n')
print('\n')