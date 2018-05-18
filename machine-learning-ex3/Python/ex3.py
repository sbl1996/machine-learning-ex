## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from display_data import display_data
from lr_cost_function import lr_cost_function
from one_vs_all import one_vs_all
from predict_one_vs_all import predict_one_vs_all
## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y'].flatten()
m = X.shape[0]

# Randomly select 100 data points to display
sel = X[np.random.choice(m, 100, replace=None)]
display_data(sel)

print('Program paused. Press enter to continue.\n')
input()

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.append(np.ones((5,1)), np.arange(1,16).reshape(5,3,order='F') / 10, axis=1)
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)

print('\nCost: %s\n' % J)
print('Expected cost: 2.534819\n')
print('Gradients:\n')
print(' %s \n' % grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

print('Program paused. Press enter to continue.\n')
input()

## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lamb = 0.1
all_theta = one_vs_all(X, y, num_labels, lamb)

print('Program paused. Press enter to continue.\n')
input()

## ================ Part 3: Predict for One-Vs-All ================

p = predict_one_vs_all(all_theta, X)

print('\nTraining Set Accuracy: %s\n' % (np.double(p == y).mean() * 100))

