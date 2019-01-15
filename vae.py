from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K


def encoder(x):
    conv_0  = Conv2D(filters=32,
                     kernel_size=(3,3),
                     strides=(2,2),
                     padding='same',
                     activation='relu')(x)

    conv_1  = Conv2D(filters=64,
                     kernel_size=(3,3),
                     strides=(2,2),
                     padding='same',
                     activation='relu')(conv_0)

    flat    = Flatten()(conv_1)
    hidden  = Dense(units=128, activation='relu')(flat)

    # Mean and Log Variance of latent space
    mean    = Dense(units=4)(hidden)
    log_var = Dense(units=4)(hidden)
    return mean, log_var


def sampler(inputs):
    mean, log_var = inputs
    batch = K.shape(mean)[0]
    dim = K.shape(mean)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mean + K.exp(log_var) * eps


def decoder(z):
    hidden    = Dense(units=128, activation='relu')(z)
    up_sample = Dense(units=4096, activation='relu')(hidden)
    reshape   = Reshape((8,8,64))(up_sample)

    deconv_0  = Conv2DTranspose(filters=32,
                                kernel_size=(3,3),
                                strides=(2,2),
                                padding='same',
                                activation='relu')(reshape)

    deconv_1  = Conv2DTranspose(filters=3,
                                kernel_size=(3,3),
                                strides=(2,2),
                                padding='same',
                                activation='relu')(deconv_0)
    return deconv_1

