from keras.layers import Input
from keras.callbacks import TensorBoard
from keras.models import load_model

from helper import *
from vae import *

if __name__ == '__main__':
    _, x_test = load_dataset('cifar10')

    model = load_model('./models/trained_vae.h5')

    x = x_test[0]
    x = x[None,...]
    x_prime = model.predict(x)

    compare_result(x_prime, x)
    rmse(x_prime, x)
