import cv2
from keras.layers import Input
from keras.callbacks import TensorBoard
from keras.models import load_model

from helper import *
from vae import *

if __name__ == '__main__':
    model = load_model('./models/trained_vae.h5')

    x = cv2.imread('./dog.jpg')
    x = cv2.resize(x, (32,32))
    x = x[:,:,::-1]
    x = x / 255.0
    x = x[None,...]

    x_prime = model.predict(x)

    rmse(x_prime, x)
    compare_result(x_prime, x)
