import numpy as np
import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose
from tensorflow.layers import batch_normalization as BN
from tensorflow.layers import flatten
from tensorflow.layers import dense
from tensorflow.nn import relu, leaky_relu, sigmoid

from config import *

def ACT(inputs, act_fn):
    if act_fn == 'relu':
        act = relu(inputs)
    elif act_fn == 'lrelu':
        act = leaky_relu(inputs)
    elif act_fn == 'sigmoid':
        act = sigmoid(inputs)
    return act


def CONV(inputs, filters, kernel_size, strides, padding, is_transpose):
    if is_transpose:
        conv = conv2d_transpose(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding)
    else:
        conv = conv2d(inputs=inputs,
                      filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding)
    return conv


def CONV_BN_ACT(inputs, filters, kernel_size, strides, padding, act_fn, is_training, is_transpose):
    conv = CONV(inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                is_transpose=is_transpose)

    norm = BN(inputs=conv, training=is_training)
    act  = ACT(inputs=norm, act_fn=act_fn)
    return act


def Encoder(inputs, is_training):
    with tf.variable_scope('Encoder'):
        conv_0 = CONV_BN_ACT(inputs=inputs,
                             filters=32,
                             kernel_size=[3,3],
                             strides=[1,1],
                             padding='same',
                             act_fn='lrelu',
                             is_training=is_training,
                             is_transpose=False)

        conv_1 = CONV_BN_ACT(inputs=conv_0,
                             filters=64,
                             kernel_size=[3,3],
                             strides=[2,2],
                             padding='same',
                             act_fn='lrelu',
                             is_training=is_training,
                             is_transpose=False)

        conv_2 = CONV_BN_ACT(inputs=conv_1,
                             filters=128,
                             kernel_size=[3,3],
                             strides=[2,2],
                             padding='same',
                             act_fn='lrelu',
                             is_training=is_training,
                             is_transpose=False)

        flat = flatten(conv_2)

        z_mean = dense(inputs=flat, units=z_dim)
        z_log_var = dense(inputs=flat, units=z_dim)
        return z_mean, z_log_var


def Decoder(z, is_training):
    with tf.variable_scope('Decoder'):
        upsample = dense(inputs=z,
                         units=8192,
                         activation=relu)

        reshaped = tf.reshape(upsample, [-1,8,8,128])

        tconv_0 = CONV_BN_ACT(inputs=reshaped,
                              filters=64,
                              kernel_size=[3,3],
                              strides=[2,2],
                              padding='same',
                              act_fn='lrelu',
                              is_training=is_training,
                              is_transpose=True)

        tconv_1 = CONV_BN_ACT(inputs=tconv_0,
                              filters=32,
                              kernel_size=[3,3],
                              strides=[2,2],
                              padding='same',
                              act_fn='lrelu',
                              is_training=is_training,
                              is_transpose=True)

        tconv_2 = CONV_BN_ACT(inputs=tconv_1,
                              filters=3,
                              kernel_size=[3,3],
                              strides=[1,1],
                              padding='same',
                              act_fn='sigmoid',
                              is_training=is_training,
                              is_transpose=True)
        return tconv_2


def Vae(inputs, is_training=True):
    # Encode
    z_mean, z_log_var = Encoder(inputs, is_training)
    # Sample
    epsilon = tf.random_normal(tf.shape(z_mean))
    z = z_mean + tf.exp(z_log_var) * epsilon if (
        is_training) else z_mean
    # Decode
    y = Decoder(z, is_training)
    return y, z_mean, z_log_var

