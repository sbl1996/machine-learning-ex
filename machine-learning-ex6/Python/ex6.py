## Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import classification_report

from plot_data import plot_data
from visualize_boundary import visualize_boundary
from gaussian_kernel import gaussian_kernel
from dataset3_params import dataset3_params

plt.ion()

## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data1: 
# You will have X, y in your environment
data = loadmat('ex6data1.mat')
X = data['X']
y = data['y'].ravel()

# Plot training data
fig, ax = plt.subplots()
plot_data(X, y, ax)

print('Program paused. Press enter to continue.\n')
input()

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svm.LinearSVC(C=C, tol=1e-3, max_iter=20)
model.fit(X, y)
visualize_boundary(X, y, model)

print('Program paused. Press enter to continue.\n')
input()

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \
\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim))

print('Program paused. Press enter to continue.\n')
input()

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data2:
# You will have X, y in your environment
data = loadmat('ex6data2.mat')
X = data['X']
y = data['y'].ravel()
# Plot training data
fig, ax = plt.subplots()
plot_data(X, y, ax)

print('Program paused. Press enter to continue.\n')
input()

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')

# SVM Parameters
C = 1
gamma = 50

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
model.fit(X, y)
visualize_boundary(X, y, model)

print('Program paused. Press enter to continue.\n')
input()

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...\n')

# Load from ex6data3: 
# You will have X, y in your environment
data = loadmat('ex6data3.mat')
X = data['X']
y = data['y'].ravel()
Xval = data['Xval']
yval = data['yval'].ravel()

# Plot training data
fig, ax = plt.subplots()
plot_data(X, y, ax)

print('Program paused. Press enter to continue.\n')
input()

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 

# Try different SVM Parameters here
C, gamma, model = dataset3_params(X, y)
y_true, y_pred = yval, model.predict(Xval)
print(classification_report(y_true, y_pred))

visualize_boundary(X, y, model)

print('Program paused. Press enter to continue.\n')
input()

