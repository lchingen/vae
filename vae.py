from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import ReLU, LeakyReLU, PReLU, BatchNormalization
from keras.layers import Activation
from keras.layers import Layer
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import backend as K


def ACT(input, act_type, alpha=0):
    if act_type == 'prelu':
        act = PReLU(alpha=alpha)(input)
    elif act_type == 'lrelu':
        act = LeakyReLU(alpha=alpha)(input)
    elif act_type == 'relu':
        act = ReLU()(input)
    elif act_type == 'tanh':
        act = Activation('tanh')(input)
    elif act_type == 'sigmoid':
        act = Activation('sigmoid')(input)
    return act


def CONV_BN_RELU(input, filters, kernel_size, strides, padding, act_type, alpha):
    conv  = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   activation='linear')(input)

    norm  = BatchNormalization()(conv)
    act   = ACT(input=norm,
                act_type=act_type,
                alpha=alpha)

    return act


def TCONV_BN_RELU(input, filters, kernel_size, strides, padding, act_type, alpha):
    conv  = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation='linear')(input)
 
    norm  = BatchNormalization()(conv)
    act   = ACT(input=norm,
                act_type=act_type,
                alpha=alpha)

    return act



def FC_RELU(input, units, act_type, alpha):
    fc  = Dense(units=units, activation='linear')(input)

    act = ACT(input=conv,
              act_type=act_type,
              alpha=alpha)
    return act


def Encoder(x):
    conv_0  = CONV_BN_RELU(input=x,
                           filters=32,
                           kernel_size=(3,3),
                           strides=(1,1),
                           padding='same',
                           act_type='lrelu',
                           alpha=0.125)

    conv_1  = CONV_BN_RELU(input=conv_0,
                           filters=64,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           act_type='lrelu',
                           alpha=0.125)

    conv_2  = CONV_BN_RELU(input=conv_1,
                           filters=128,
                           kernel_size=(3,3),
                           strides=(2,2),
                           padding='same',
                           act_type='lrelu',
                           alpha=0.125)


    flat    = Flatten()(conv_2)
    # Mean and Log Variance of latent space
    mean    = Dense(units=32)(flat)
    log_var = Dense(units=32)(flat)
    return mean, log_var


def Decoder(z):
    up_sample = Dense(units=8192, activation='relu')(z)
    reshape   = Reshape((8,8,128))(up_sample)

    tconv_0   = TCONV_BN_RELU(input=reshape,
                              filters=64,
                              kernel_size=(3,3),
                              strides=(2,2),
                              padding='same',
                              act_type='lrelu',
                              alpha=0.125)

    tconv_1   = TCONV_BN_RELU(input=tconv_0,
                              filters=32,
                              kernel_size=(3,3),
                              strides=(2,2),
                              padding='same',
                              act_type='lrelu',
                              alpha=0.125)

    tconv_2   = TCONV_BN_RELU(input=tconv_1,
                              filters=3,
                              kernel_size=(3,3),
                              strides=(1,1),
                              padding='same',
                              act_type='sigmoid',
                              alpha=0.125)

    return tconv_2

def Sampler(inputs):
    mean, log_var = inputs
    batch = K.shape(mean)[0]
    dim = K.shape(mean)[1]
    eps = K.random_normal(shape=(batch, dim))
    return mean + K.exp(0.5*log_var) * eps

