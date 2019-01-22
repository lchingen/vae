import numpy as np
import tensorflow as tf
from vae_celeb import *
from config import *

def model_fn(features, mode):
    # Instantiate model
    if type(features) is dict:
        x = features['x']
        z = features['z']
    else:
        x = features
        z = tf.constant(0)

    y, z_mean, z_log_var = Vae(x, z, FLAGS.mode)

    # Loss function
    rec_loss = tf.reduce_sum(tf.squared_difference(x, y)) / rec_norm
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) / kl_norm
    total_loss = tf.reduce_mean(rec_loss + beta*kl_loss)

    # Outputs
    predictions = {'x': x, 'y': y, 'mu':z_mean, 'sigma':z_log_var}

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
            summary_hook = tf.train.SummarySaverHook(save_steps=FLAGS.save_steps,
                                                     output_dir=FLAGS.export_path,
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
    x  = tf.placeholder(dtype=tf.float32, shape=[None] + input_dim, name='x')
    z  = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='z')
    features={'x':x, 'z':z}
    #return tf.estimator.export.TensorServingInputReceiver(features, features)
    return tf.estimator.export.ServingInputReceiver(features, features)


def export_tf_model(export_path):
    estimator = tf.estimator.Estimator(model_fn, export_path)
    estimator.export_saved_model(FLAGS.export_path + '/frozen_pb', serving_input_fn)
