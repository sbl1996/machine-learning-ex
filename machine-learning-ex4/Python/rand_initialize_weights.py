import numpy as np

def rand_initialize_weights(L_in, L_out):
 epsilon_init = 0.12
 W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
 return W
