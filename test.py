import numpy as np
import random
import argparse
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import *
from helper import *
from utils import create_dataset
from vae_celeb import *

tf.enable_eager_execution()


def model_fn(features, mode):
    # Instantiate model
    y, z_mean, z_log_var = Vae(features, is_training=False)

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


def serving_input_fn():
    # Export estimator as a tf serving API
    features  = tf.placeholder(dtype=tf.float32, shape=[None]+input_dim, name='features')
    return tf.estimator.export.TensorServingInputReceiver(features, features)


def export_tf_model(export_path):
    estimator = tf.estimator.Estimator(model_fn, './logs')
    estimator.export_saved_model(export_path, serving_input_fn)


if __name__ == '__main__':
    # Find latest frozen pb
    export_path = './logs/frozen_pb'
    export_tf_model(export_path)
    subdirs = [x for x in Path(export_path).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)
    dataset = create_dataset(path='./db/test.tfrecords',
                             buffer_size=25,
                             batch_size=25,
                             num_epochs=1)

    iterator = dataset.make_one_shot_iterator()

    # Eager execution for obtaining batch data from dataset
    value = iterator.get_next().numpy()

    # Put in an input dict as per predict_fn input definition
    x = {'input': value}

    # Make predictions and fetch results from output dict
    predictions = predict_fn(x)
    x_src = predictions['x_src']
    x_gen = predictions['x_gen']

    # Show all source v.s. generated results
    show_all(x_src, x_gen, x_src.shape[0])
