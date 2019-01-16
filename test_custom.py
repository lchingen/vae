import os
import sys
import cv2
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

    # Read custom image
    x = cv2.imread('./dog.jpg')
    x = cv2.resize(x, (32,32))
    x = x[:,:,::-1]
    x = x / 255.0
    x = x[None,...]

    # Test generation
    x_gen = vae_model.predict(x)
    compare_result(x, x_gen)
