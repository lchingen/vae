import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import *
from helper import *
from utils import *
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

    # Read image
    x  = load_img('./imgs/end.jpg')[None] / 255.0
    dict_in = {'x': x, 'z': np.zeros(z_dim)[None]}

    # Make predictions and fetch results from output dict
    z = predict_fn(dict_in)['mu']
    z_shift = np.roll(z, 1, axis = 1)

    z_val = np.vstack((z, z_shift))

    dict_in = {'x': np.zeros(input_dim)[None], 'z': z_val}
    y, y_shift = predict_fn(dict_in)['y']

    compare(y, y_shift)

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', None, 'TRAIN/TEST')
    tf.app.run()
