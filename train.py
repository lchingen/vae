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
    y, z_mean, z_log_var = Vae(features, is_training=True)

    # Loss function
    rec_loss = tf.reduce_sum(tf.squared_difference(features, y)) / rec_norm
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)) / kl_norm
    total_loss = tf.reduce_mean(rec_loss + beta*kl_loss)

    # Mode selection
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Set up logging hooks
        tf.summary.scalar('rec_loss', rec_loss)
        tf.summary.scalar('kl_loss', kl_loss)
        summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                 output_dir='./logs',
                                                 summary_op=tf.summary.merge_all())

        # Set up optimizer
        optimizer = tf.train.AdamOptimizer(0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=total_loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          train_op=train_op,
                                          training_hooks=[summary_hook])

    '''
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'reconstruction_loss': tf.metrics.root_mean_squared_error(labels=features, predictions=predictions['output'])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    '''


def train(num_epochs, batch_size, dataset):
    x_train, _ = load_dataset(dataset)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x_train,
                                                        shuffle=True,
                                                        batch_size=batch_size,
                                                        num_epochs=num_epochs)
    with tf.Session() as sess:
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')
        estimator.train(input_fn=train_input_fn)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dataset')
    args = parser.parse_args()

    train(args.num_epochs, args.batch_size, args.dataset)

    '''
    elif args.mode == 'visual':
        estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='./logs')
        visualize('conv0/kernel')

    else:
        print('Please enter the operation mode (--mode train/test/visual)')
    '''
