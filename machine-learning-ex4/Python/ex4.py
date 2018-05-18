## Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

from nn_cost_function import nn_cost_function
from display_data import display_data
from sigmoid_gradient import sigmoid_gradient
from rand_initialize_weights import rand_initialize_weights
from check_nn_gradients import check_nn_gradients
from predict import predict
## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

plt.ion()

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = loadmat('ex4data1.mat')
RX = data['X']
Ry = data['y']
m = RX.shape[0]

X = RX
y = Ry.ravel()
# Randomly select 100 data points to display
sel = X[np.random.choice(m, 100, replace=None)]

display_data(sel)
plt.pause(1)

print('Program paused. Press enter to continue.\n')
input()


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weights = loadmat('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

# Unroll parameters 
nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
lamb = 0

J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lamb)

print('Cost at parameters (loaded from ex4weights): %s \
\n(this value should be about 0.287629)\n' % J)

print('\nProgram paused. Press enter to continue.\n')
input()

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
lamb = 1

J, grad = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, \
                     num_labels, X, y, lamb)

print('Cost at parameters (loaded from ex4weights): %s \
\n(this value should be about 0.383770)\n' % J)

print('Program paused. Press enter to continue.\n')
input()


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print('%s ' % g)
print('\n\n')

print('Program paused. Press enter to continue.\n')
input()

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
check_nn_gradients()

print('\nProgram paused. Press enter to continue.\n')
input()


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lamb = 3
check_nn_gradients(lamb)

# Also output the costFunction debugging values
debug_J  = nn_cost_function(nn_params, input_layer_size, \
                          hidden_layer_size, num_labels, X, y, lamb)[0]

print('\n\nCost at (fixed) debugging parameters (w/ lambda = %s): %s \
\n(for lambda = 3, this value should be about 0.576051)\n\n' % (lamb, debug_J))

print('Program paused. Press enter to continue.\n')
input()


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.

#  You should also try different values of lambda
lamb = 1

# Create "short hand" for the cost function to be minimized
cost_function = lambda p: nn_cost_function(p, \
                          input_layer_size, \
                          hidden_layer_size, \
                          num_labels, X, y, lamb)

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
nn_params, cost, _ = fmin_tnc(cost_function, initial_nn_params)

# Obtain Theta1 and Theta2 back from nn_params
s1 = input_layer_size
s2 = hidden_layer_size
s3 = num_labels
Theta1 = nn_params[:s2 * (s1 + 1)].reshape(s2, s1 + 1)
Theta2 = nn_params[s2 * (s1 + 1):].reshape(s3, s2 + 1)

print('Program paused. Press enter to continue.\n')
input()


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')

display_data(Theta1[:, 1:])

print('\nProgram paused. Press enter to continue.\n')
input()

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: %s\n' % (np.double(pred == y).mean() * 100))