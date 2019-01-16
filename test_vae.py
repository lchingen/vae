import os
import sys
from keras.layers import Input
from keras.callbacks import TensorBoard
from keras.models import load_model

from helper import *
from vae import *

if __name__ == '__main__':
    # Instantiate model
    x = Input(shape=(32,32,3))
    mean, log_var = encoder(x)
    z = Lambda(sampler, output_shape=(2,))([mean, log_var])
    y = decoder(z)

    vae_model = Model(x, y)

    # Compute Loss
    reconstruction_loss = K.sqrt(K.mean(x - y)**2)
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var))
    total_loss = reconstruction_loss + kl_loss

    vae_model.add_loss(total_loss)
    vae_model.compile(optimizer='adam')
    vae_model.summary()

    if os.path.exists('./models/trained_vae_weights.h5'):
        vae_model.load_weights('./models/trained_vae_weights.h5')
    else:
        print('Weights not found...')
        sys.exit(0)

    # Fetch dataset
    _, x_test = load_dataset('cifar10')

    # Test inference
    x = x_test[100]
    #x = x[:,:,::-1]
    x = x[None,...]
    generated = vae_model.predict(x)

    compare_result(x, generated)
    rmse(generated, x)
