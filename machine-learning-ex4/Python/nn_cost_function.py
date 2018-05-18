import numpy as np

from sigmoid import sigmoid
def nn_cost_function(nn_params,
                     input_layer_size,
                     hidden_layer_size,
                     num_labels,
                     X, y, lamb):
  #NNCOSTFUNCTION Implements the neural network cost function for a two layer
  #neural network which performs classification
  #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  #   X, y, lambda) computes the cost and gradient of the neural network. The
  #   parameters for the neural network are "unrolled" into the vector
  #   nn_params and need to be converted back into the weight matrices. 
  # 
  #   The returned parameter grad should be a "unrolled" vector of the
  #   partial derivatives of the neural network.
  #

  # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  # for our 2 layer neural network
  s1 = input_layer_size
  s2 = hidden_layer_size
  s3 = num_labels
  Theta1 = nn_params[:s2 * (s1 + 1)].reshape(s2, s1 + 1)
  Theta2 = nn_params[s2 * (s1 + 1):].reshape(s3, s2 + 1)

  # Setup some useful variables
  m = X.shape[0]

  def gen_y(v):
    v1 = np.zeros((num_labels,))
    v1[v[0] - 1] = 1
    return v1

  y = np.apply_along_axis(gen_y, 1, y.reshape(y.shape[0], 1))
           
  # You need to return the following variables correctly 
  J = 0
  Theta1_grad = np.zeros(Theta1.shape)
  Theta2_grad = np.zeros(Theta2.shape)

  # ====================== YOUR CODE HERE ======================
  # Instructions: You should complete the code by working through the
  #               following parts.
  #
  # Part 1: Feedforward the neural network and return the cost in the
  #         variable J. After implementing Part 1, you can verify that your
  #         cost function computation is correct by verifying the cost
  #         computed in ex4.m
  #
  # Part 2: Implement the backpropagation algorithm to compute the gradients
  #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  #         Theta2_grad, respectively. After implementing Part 2, you can check
  #         that your implementation is correct by running checkNNGradients
  #
  #         Note: The vector y passed into the function is a vector of labels
  #               containing values from 1..K. You need to map this vector into a 
  #               binary vector of 1's and 0's to be used with the neural network
  #               cost function.
  #
  #         Hint: We recommend implementing backpropagation using a for-loop
  #               over the training examples if you are implementing it for the 
  #               first time.
  #
  # Part 3: Implement regularization with the cost function and gradients.
  #
  #         Hint: You can implement this around the code for
  #               backpropagation. That is, you can compute the gradients for
  #               the regularization separately and then add them to Theta1_grad
  #               and Theta2_grad from Part 2.
  #
  K = num_labels
  a1 = np.append(np.ones((m, 1)), X, axis=1)
  a2 = np.append(np.ones((m, 1)), sigmoid(a1 @ Theta1.T), axis=1)
  a3 = sigmoid(a2 @ Theta2.T)
  h = a3
  J = (y * np.log(h) + (np.ones(y.shape) - y) * np.log(np.ones(h.shape) - h)).sum() / (-m)
  reg = (lamb / (2 * m)) * ((Theta1[:, 1:] ** 2).sum() + (Theta2[:, 1:] ** 2).sum())
  J += reg

  delta3 = a3 - y
  delta2 = ((delta3 @ Theta2) * a2 * (np.ones(a2.shape) - a2))[:, 1:]
  Delta2 = delta3.T @ a2
  Delta1 = delta2.T @ a1
  D2 = np.zeros(Theta2.shape)
  D2[:, 1:] = (Delta2[:, 1:] + lamb * Theta2[:, 1:]) / m
  D2[:, 0] = Delta2[:, 0] / m
  D1 = np.zeros(Theta1.shape)
  D1[:, 1:] = (Delta1[:, 1:] + lamb * Theta1[:, 1:]) / m
  D1[:, 0] = Delta1[:, 0] / m
  
  Theta2_grad = D2
  Theta1_grad = D1





  # -------------------------------------------------------------

  # =========================================================================

  # Unroll gradients
  grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()))
  
  return J, grad

