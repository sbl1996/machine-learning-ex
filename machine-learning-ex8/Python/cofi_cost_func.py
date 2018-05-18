import numpy as np
from scipy.linalg import norm

def cofi_cost_func(params, Y, R, num_users, num_movies,
                   num_features, lamda):
  
  X = params[:num_movies*num_features].reshape(num_movies, num_features)
  Theta = params[num_movies*num_features:].reshape(num_users, num_features)

  temp = X @ Theta.T - Y
  J = (temp ** 2 * R).sum() / 2 + (lamda / 2) * (norm(Theta) ** 2 + norm(X) ** 2)
  X_grad = temp * R @ Theta + lamda * X
  Theta_grad = (temp * R).T @ X + lamda * Theta

  grad = np.r_[X_grad.ravel(), Theta_grad.ravel()]
  return J, grad