import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import *
from helper import *
from utils import create_dataset
from model_fn import *

tf.enable_eager_execution()

def main(unused_argv):
    # Export model_fn to only use decoder
    export_tf_model(FLAGS.export_path)

    # Find latest frozen pb
    subdirs = [x for x in Path(FLAGS.export_path + '/frozen_pb').iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)

    # Eager execution for obtaining batch data from dataset
    N = 64
    x_val = np.ones(input_dim)[None]
    z_val = np.ones([64, z_dim])

    t = np.linspace(0.05, 0.95, N)
    #for ii in range(z_dim):
    #    z_val[:, ii] = t
    for ii in np.arange(20):
        z_val[:, ii] = t

    # Put in an input dict as per predict_fn input definition
    dict_in = {'x': x_val, 'z': z_val}

    # Make predictions and fetch results from output dict
    predictions = predict_fn(dict_in)
    x = predictions['x']
    y = predictions['y']
    z = predictions['z']

    show_all(y, N)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', None, 'TRAIN/TEST')
    tf.app.run()
