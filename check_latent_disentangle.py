import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path
from matplotlib.pyplot import plot as plt

from config import *
from helper import *
from utils import *
from model_fn import *


def shift(m_in, offset=0, dir='lr'):
    # Generate permutation matrix
    p = np.eye(m_in.shape[0])
    p = np.roll(p, offset, axis=0).astype('int')

    # Horizontal shift (left/right = +/- offset)
    if dir == 'lr':
        m_in = np.transpose(m_in, (2, 0, 1))
        x = np.matmul(m_in, p)
        return np.transpose(x, (1, 2, 0)).astype('uint8')

    # Vertical shift (down/up = +/- offset)
    if dir == 'ud':
        x = np.matmul(m_in.T, p).T
        return x.astype('uint8')


def plot_lines(data):
    fig = plt.figure()
    ax = plt.axes()

    x = np.linspace(0, len(data)-1, len(data[0]))
    for ii in range(data.shape[0]):
        ax.plot(x, data[ii])
    plt.show()

def main(unused_argv):
    # Export model_fn to only use decoder
    export_tf_model(FLAGS.export_path)

    # Find latest frozen pb
    subdirs = [x for x in Path(FLAGS.export_path + '/frozen_pb').iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)

    # Read image
    x  = load_img('./imgs/end.jpg')
    x_shift = shift(x, offset=5, dir='lr')
    x = x[None, ...] /255.0
    x_shift = x_shift[None, ...] /255.0

    x_val = np.vstack((x, x_shift))

    dict_in = {'x': x_val, 'z': np.zeros(z_dim)[None]}

    # Make predictions and fetch results from output dict
    z = predict_fn(dict_in)['mu']
    plot_lines(z)

    #z_shift = np.roll(z, 20, axis = 1)
    #z_val = np.vstack((z, z_shift))

    #dict_in = {'x': np.zeros(input_dim)[None], 'z': z_val}
    #y, y_shift = predict_fn(dict_in)['y']

    #compare(y, y_shift)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', None, 'TRAIN/TEST')
    tf.app.run()
