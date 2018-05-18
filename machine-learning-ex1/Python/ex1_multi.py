import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normal_equation import normal_equation
from feature_normalize import feature_normalize
from gradient_descent_h import gradient_descent_h

print('Loading data ...\n')

data = pd.read_csv('ex1data2.txt', header=None).as_matrix()
X = data[:, :2]
y = data[:, [-1]]
m = len(y)

print('Program paused. Press enter to continue.\n')
input()

print('Normalizing Features ...\n')

mu = np.apply_along_axis(lambda v: v.mean(), 0, X)
sigma = np.apply_along_axis(lambda v: v.std(), 0, X)
X = feature_normalize(X)

X = np.append(np.ones((m,1)), X, 1)

print('Running gradient descent ...\n')

alpha = 0.01
num_iters = 400

theta = np.zeros((3, 1))
theta, J_history = gradient_descent_h(X, y, theta, alpha, num_iters)

% Plot the convergence graph
fig, ax = plt.subplots()
ax.plot(J_history, '-b', linewidth=2)
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Cost J')

print('Theta computed from gradient descent: \n')
print(' %s \n' % theta)
print('\n')

example_norm = np.append([1], (np.array([1650, 3]) - mu) / sigma, 0)
price = (example_norm @ theta)[0]

print('Predicted price of a 1650 sq-ft, 3 br house\
(using gradient descent):\n $%f\n' % price)

print('Program paused. Press enter to continue.\n')
input()

print('Solving with normal equations...\n')


data = pd.read_csv('ex1data2.txt', header=None).as_matrix()
X = data[:, :2]
y = data[:, [-1]]
m = len(y)

X = np.append(np.ones((m,1)), X, 1)

theta = normal_equation(X, y)

print('Theta computed from the normal equations: \n')
print(' %f \n' % theta)
print('\n')


price = (np.array([1, 1650, 3]) @ theta)[0]

print('Predicted price of a 1650 sq-ft, 3 br house \
         (using normal equations):\n $%f\n' % price)

