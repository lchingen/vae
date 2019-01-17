import numpy as np

# Training configurations
input_dim = [32, 32, 3]
z_dim     = 100
beta      = 0.1
rec_norm  = np.prod(input_dim)
kl_norm   = z_dim
