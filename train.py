import numpy as np
import random
import argparse
import tensorflow as tf

from tensorflow.python.estimator.inputs import numpy_io

from vae_celeb import *
from config import *
from helper import *
from utils import create_dataset

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, mode):
    # Instantiate model
    y, z_mean, z_log_var = Vae(features, is_training=True)

    # Loss function
    rec_loss = tf.reduce_sum(tf.squared_difference(features, y)) / rec_norm
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) / kl_norm
    total_loss = tf.reduce_mean(rec_loss + beta*kl_loss)

    # Outputs
    predictions = {'x_src': features, 'x_gen': y}

    # Mode selection
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            # Set up logging hooks
            tf.summary.scalar('rec_loss', rec_loss)
            tf.summary.scalar('kl_loss', kl_loss)
            summary_hook = tf.train.SummarySaverHook(save_steps=5,
                                                     output_dir='./logs',
                                                     summary_op=tf.summary.merge_all())

            # Set up optimizer
            optimizer = tf.train.AdamOptimizer(learn_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss=total_loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=total_loss,
                                              train_op=train_op,
                                              training_hooks=[summary_hook])
        else:
            raise NotImplementedError()


def train_input_fn_from_tfr():
    return lambda: create_dataset(path='./db/train.tfrecords',
                                  buffer_size=buffer_size,
                                  batch_size=batch_size,
                                  num_epochs=num_epochs)


def train_input_fn_from_numpy():
    x_train, _ = load_dataset(dataset)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x_train,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        num_epochs=num_epochs)


if __name__ == '__main__':
    with tf.Session() as sess:
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')
        estimator.train(train_input_fn_from_tfr())
        sess.close()
