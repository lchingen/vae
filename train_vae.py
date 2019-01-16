import os
from keras.layers import Input
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam

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
    if not os.path.exists('./models'):
        os.makedirs('models')
    vae_model.save_weights('./models/trained_vae_weights.h5')

    # Test inference
    x = x_test[100]
    #x = x[:,:,::-1]
    x = x[None,...]
    generated = vae_model.predict(x)

    compare_result(x, generated)
    rmse(generated, x)
