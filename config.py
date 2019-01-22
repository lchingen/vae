import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Training configurations
input_dim   = [64, 64, 3]
learn_rate  = 0.0005
z_dim       = 256
beta        = 0.01
rec_norm    = np.prod(input_dim)
kl_norm     = z_dim

tf.app.flags.DEFINE_string('dataset',     'celeb-face', '')
tf.app.flags.DEFINE_string('export_path', './logs', '')
tf.app.flags.DEFINE_string('train_path',  './db/train.tfrecords', '')
tf.app.flags.DEFINE_string('vld_path',    './db/train.tfrecords', '')
tf.app.flags.DEFINE_string('test_path',   './db/train.tfrecords', '')

tf.app.flags.DEFINE_integer('num_epochs',  10, '')
tf.app.flags.DEFINE_integer('batch_size',  128, '')
tf.app.flags.DEFINE_integer('buffer_size', 10000, '')
tf.app.flags.DEFINE_integer('save_steps',  10, '')
