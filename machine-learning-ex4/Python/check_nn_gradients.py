import numpy as np
from scipy import linalg

from debug_initialize_weights import debug_initialize_weights
from nn_cost_function import nn_cost_function
from compute_numerical_gradient import compute_numerical_gradient

def check_nn_gradients(lamb = 0):
  # CHECKNNGRADIENTS Creates a small neural network to check the
  # backpropagation gradients
  #   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
  #   backpropagation gradients, it will output the analytical gradients
  #   produced by your backprop code and the numerical gradients (computed
  #   using computeNumericalGradient). These two gradient computations should
  #   result in very similar values.
  #

  input_layer_size = 3
  hidden_layer_size = 5
  num_labels = 3
  m = 5

  # We generate some 'random' test data
  Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
  Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)
  # Reusing debugInitializeWeights to generate X
  X = debug_initialize_weights(m, input_layer_size - 1)
  y = 1 + (np.arange(1, m + 1) % num_labels)

  # Unroll parameters
  nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

  # Short hand for cost function
  cost_func = lambda p: nn_cost_function(p, input_layer_size, hidden_layer_size, \
                        num_labels, X, y, lamb)

  cost, grad = cost_func(nn_params)
  numgrad = compute_numerical_gradient(cost_func, nn_params)

  # Visually examine the two gradient computations.  The two columns
  # you get should be very similar. 
  print(np.stack((numgrad, grad), axis=1))
  print('The above two columns you get should be very similar.\n \
  (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

  # Evaluate the norm of the difference between two solutions.  
  # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
  # in computeNumericalGradient.m, then diff below should be less than 1e-9
  diff = linalg.norm(numgrad - grad) / linalg.norm(numgrad + grad)

  print('If your backpropagation implementation is correct, then \n \
  the relative difference will be small (less than 1e-9). \n \
  \nRelative Difference: %g\n' % diff)
