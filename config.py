import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Training configurations
input_dim   = [64, 64, 3]
learn_rate  = 0.001
z_dim       = 256
rec_norm    = np.prod(input_dim)
kl_norm     = z_dim
beta        = 0.1 * (rec_norm/kl_norm)

tf.app.flags.DEFINE_string('dataset',     'celeb-face', '')
tf.app.flags.DEFINE_string('export_path', './logs', '')
tf.app.flags.DEFINE_string('train_path',  './db/train.tfrecords', '')
tf.app.flags.DEFINE_string('vld_path',    './db/train.tfrecords', '')
tf.app.flags.DEFINE_string('test_path',   './db/train.tfrecords', '')

tf.app.flags.DEFINE_integer('num_epochs',  5, '')
tf.app.flags.DEFINE_integer('batch_size',  128, '')
tf.app.flags.DEFINE_integer('buffer_size', 50000, '')
tf.app.flags.DEFINE_integer('save_steps',  10, '')
