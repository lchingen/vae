import numpy as np

# Training configurations
input_dim   = [64, 64, 3]
learn_rate  = 0.0005
z_dim       = 128
beta        = 0.01
rec_norm    = np.prod(input_dim)
kl_norm     = z_dim

dataset     = 'celeb-face'
train_path  = './db/train.tfrecords'
num_epochs  = 15
batch_size  = 128
buffer_size = 1000
