import numpy as np

def email_features(word_indices):
  n = 1899
  x = np.zeros((1899,)) 
  for i in word_indices:
    x[i - 1] = 1
  return x