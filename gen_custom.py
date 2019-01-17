import numpy as np
import cv2
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

def infer(x, dataset):
    x_src = x
    pred_input_fn  = tf.estimator.inputs.numpy_input_fn(x_src,
                                                        shuffle=False,
                                                        batch_size=1,
                                                        num_epochs=1)
    with tf.Session() as sess:
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')
        prediction = list(estimator.predict(input_fn=pred_input_fn))
        sess.close()

    x_gen = prediction[0]
    compare_result(x_src, x_gen)


if __name__ == '__main__':
    x = cv2.imread('./imgs/dog.jpg')
    x = cv2.resize(x, (32,32))
    x = x[:,:,::-1]
    x = x / 255.0
    x = x[None,...]

    infer(x, 'cifar10')
