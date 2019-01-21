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
    dataset = create_dataset(path=FLAGS.test_path,
                             buffer_size=25,
                             batch_size=25,
                             num_epochs=1)

    iterator = dataset.make_one_shot_iterator()

    # Eager execution for obtaining batch data from dataset
    x_val = iterator.get_next().numpy()
    z_val = np.ones([x_val.shape[0],128])*0.1

    # Put in an input dict as per predict_fn input definition
    dict_in = {'x': x_val, 'z': z_val}

    # Make predictions and fetch results from output dict
    predictions = predict_fn(dict_in)
    x = predictions['x']
    y = predictions['y']
    z = predictions['z']

    show_all(x, y, x.shape[0])

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', None, 'TRAIN/TEST')
    tf.app.run()
