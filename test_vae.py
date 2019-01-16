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
    mean, log_var = Encoder(x)
    z = Lambda(Sampler)([mean, log_var])
    y = Decoder(z)

    vae_model = Model(x, y)

    # Load trained weights
    if os.path.exists('./models/trained_vae_weights.h5'):
        vae_model.load_weights('./models/trained_vae_weights.h5')
    else:
        print('Weights not found...')
        sys.exit(0)

    # Fetch dataset
    _, x_test = load_dataset('cifar10')

    # Test generation
    test_size = 25
    x_org = x_test[:test_size]
    x_gen = vae_model.predict(x_org)

    show_all(x_org, x_gen, test_size)