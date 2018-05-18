import numpy as np

def debug_initialize_weights(fan_out, fan_in):
  
  W = np.zeros((fan_out, fan_in + 1))

  W = np.sin(np.arange(W.size)).reshape(W.shape) / 10

  return W