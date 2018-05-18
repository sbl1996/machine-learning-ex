## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

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

from scipy.io import loadmat
import numpy as np

from display_data import display_data
from predict import predict
## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

plt.ion()

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = loadmat('ex3data1.mat')
RX = data['X']
Ry = data['y'].flatten()
m = RX.shape[0]

rindices = np.random.permutation(m)
X = RX[rindices]
y = Ry[rindices]

# Randomly select 100 data points to display
sel = X[np.random.choice(m, 100, replace=None)]

display_data(sel)
plt.pause(1)

print('Program paused. Press enter to continue.\n')
input()

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weights = loadmat('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

p = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy: %s\n' % (np.double(p == y).mean() * 100))

print('Program paused. Press enter to continue.\n')
input()

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in rp:
  # Display 
  print('\nDisplaying Example Image\n')
  display_data(X[[i]])

  pred = predict(Theta1, Theta2, X[[i]])
  print('\nNeural Network Prediction: %d (digit %d)\n' % (pred, pred % 10))
    
  # Pause with quit option
  s = input('Paused - press enter to continue, q to exit:')
  if s == 'q':
    break