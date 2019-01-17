import numpy as np
import random
import argparse
import tensorflow as tf

from tensorflow.python.estimator.inputs import numpy_io
from vae import *
from config import *
from helper import *

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, mode):
    # Instantiate model
    features = tf.cast(features, tf.float32) # BN requirement
    y, z_mean, z_log_var = Vae(features, is_training=False)

    # Mode selection
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y)

def infer(sample_size, dataset):
    _, x_test = load_dataset(dataset)
    x_src = x_test[:sample_size]
    pred_input_fn  = tf.estimator.inputs.numpy_input_fn(x_src,
                                                        shuffle=False,
                                                        batch_size=1,
                                                        num_epochs=1)
    with tf.Session() as sess:
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')
        prediction = list(estimator.predict(input_fn=pred_input_fn))
        sess.close()

    x_gen = np.zeros([sample_size, 32, 32, 3])
    for ii in range(sample_size):
        x_gen[ii] = prediction[ii]

    show_all(x_src, x_gen, sample_size)


if __name__ == '__main__':
    sample_size = 25  #random.randint(0, 10000)
    infer(sample_size, 'cifar10')
