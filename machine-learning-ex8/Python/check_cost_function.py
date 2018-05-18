import numpy as np
from compute_numerical_gradient import compute_numerical_gradient
from cofi_cost_func import cofi_cost_func
from scipy.linalg import norm

def check_cost_function(lamda=0):

  X_t = np.random.rand(4, 3)
  Theta_t = np.random.rand(5, 3)

  Y = X_t @ Theta_t.T
  Y[np.random.random_sample(Y.shape) > 0.5] = 0
  R = np.zeros(Y.shape)
  R[Y != 0] = 1

  X = np.random.standard_normal(X_t.shape)
  Theta = np.random.standard_normal(Theta_t.shape)
  num_users = Y.shape[1]
  num_movis = Y.shape[0]
  num_features = Theta_t.shape[1]

  numgrad = compute_numerical_gradient(
    lambda t: cofi_cost_func(t, Y, R, num_users, num_movis, \
      num_features, lamda), np.r_[X.ravel(), Theta.ravel()])

  J, grad = cofi_cost_func(np.r_[X.ravel(), Theta.ravel()], Y, R,
    num_users, num_movis, num_features, lamda)

  print(np.c_[numgrad, grad])
  print('The above two columns you get should be very similar.\n\
(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

  diff = norm(numgrad - grad) / norm(numgrad + grad)
  print('If your cost function implementation is correct, then \n\
the relative difference will be small (less than 1e-9). \n\
\nRelative Difference: %s\n' % diff)