import numpy as np
import tensorflow as tf

from vae_celeb import *
from config import *
from helper import *
from utils import create_dataset

tf.logging.set_verbosity(tf.logging.INFO)


def train_input_fn_from_tfr():
    return lambda: create_dataset(path='./db/train.tfrecords',
                                  buffer_size=buffer_size,
                                  batch_size=batch_size,
                                  num_epochs=num_epochs)

def eval_input_fn_from_tfr():
    return lambda: create_dataset(path='./db/val.tfrecords',
                                  buffer_size=buffer_size,
                                  batch_size=64,
                                  num_epochs=1)


if __name__ == '__main__':
    with tf.Session() as sess:
        estimator  = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn_from_tfr())
        eval_spec  = tf.estimator.EvalSpec(input_fn=eval_input_fn_from_tfr())
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        sess.close()
