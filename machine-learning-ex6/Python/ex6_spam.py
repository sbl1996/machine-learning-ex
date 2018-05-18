## Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
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

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

from process_email import process_email
from email_features import email_features
from get_vocab_list import get_vocab_list

## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('\nPreprocessing sample email (emailSample1.txt)')

# Extract Features
with open('emailSample1.txt') as f:
  file_contents = f.read()
word_indices = process_email(file_contents)

# Print Stats
print('Word Indices: ')
print(' %s' % word_indices)
print('\n')

print('Program paused. Press enter to continue.')
input()

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('\nExtracting features from sample email (emailSample1.txt)')

# Extract Features
features = email_features(word_indices)

# Print Stats
print('Length of feature vector: %d' % len(features))
print('Number of non-zero entries: %d' % sum(features > 0))

print('Program paused. Press enter to continue.')
input()

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat('spamTrain.mat')
X = data['X']
y = data['y'].ravel()

print('\nTraining Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
model = svm.LinearSVC(C=C)
model.fit(X, y)

p = model.predict(X)

print('Training Accuracy: %f' % (model.score(X, y) * 100))

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
test_data = loadmat('spamTest.mat')
X_test = test_data['Xtest']
y_test = test_data['ytest']

print('\nEvaluating the trained Linear SVM on a test set ...')

print('Test Accuracy: %f' % (model.score(X_test, y_test) * 100))
input()


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtin the vocabulary list
idx = np.argsort(model.coef_)[0][::-1]
rvocab = { v: k for k, v in get_vocab_list().items() }

print('\nTop predictors of spam: ')
for i in range(15):
  print(' %-15s (%f) ' % (rvocab[idx[i]], model.coef_[0, idx[i]]))

print('\n')
print('\nProgram paused. Press enter to continue.')
input()

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'spamSample1.txt'

# Read and predict
with open(filename) as f:
  file_contents = f.read()
word_indices = process_email(file_contents)
x = email_features(word_indices)
p = model.predict([x])[0]

print('\nProcessed %s\n\nSpam Classification: %d' % (filename, p))
print('(1 indicates spam, 0 indicates not spam)\n')