from math import sqrt
from math import floor
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

def display_data(X, example_width=None):
  if not example_width:
    example_width = round(sqrt(X.shape[1]))

  m, n = X.shape
  example_height = floor(n / example_width)

  rows = floor(sqrt(m))
  cols = ceil(m / rows)

  pad = 1

  display_array = - np.ones(
    (pad + rows * (example_height + pad),
     pad + cols * (example_width + pad)))

  curr = 0
  for i in range(rows):
    for j in range(cols):
      if curr >= m:
        break

      max_val = np.abs(X[curr, :]).max()
      i_offset = pad + i * (example_height + pad)
      j_offset = pad + j * (example_width + pad)
      display_array[i_offset:i_offset + example_height,
                    j_offset:j_offset + example_width] = \
              X[curr, :].reshape(example_height, example_width, order='F') / max_val
      curr = curr + 1
    if curr >= m:
      break

  #fig, ax = plt.subplots()
  #ax.imshow(display_array)
  plt.imshow(display_array)
  plt.pause(0.1)