from keras.layers import Input
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.losses import mse
from keras.optimizers import Adam

from helper import *
from vae import *

if __name__ == '__main__':
    # Instantiate model
    x = Input(shape=(32,32,3))
    mean, log_var = encoder(x)
    z = Lambda(sampler, output_shape=(4,))([mean, log_var])
    y = decoder(z)

    vae_model = Model(x, y)

    # Compute Loss
    reconstruction_loss = K.sqrt(K.mean(x - y)**2)
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var))
    total_loss = reconstruction_loss + kl_loss

    vae_model.add_loss(total_loss)
    vae_model.compile(optimizer='adam')
    vae_model.summary()

    # Fetch dataset
    x_train, x_test = load_dataset('cifar10')

    # Train
    vae_model.fit(x_train,
                  epochs=20,
                  batch_size=64,
                  shuffle=True,
                  validation_data=(x_test, None),
                  callbacks=[TensorBoard(log_dir='./logs')])

    # Save model
    vae_model.save('./models/trained_vae.h5')
