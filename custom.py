import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from tensorflow.python.estimator.inputs import numpy_io
from test import serving_input_fn, export_tf_model
from train import model_fn
from config import *
from helper import *

tf.logging.set_verbosity(tf.logging.INFO)
tf.enable_eager_execution()


if __name__ == '__main__':
    # Find latest frozen pb
    export_path = './logs/frozen_pb'
    export_tf_model(export_path)
    subdirs = [x for x in Path(export_path).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)

    # Read image
    x = cv2.imread('./imgs/alex.jpg')
    x = cv2.resize(x, (input_dim[0], input_dim[1]),
                   interpolation=cv2.INTER_CUBIC)
    x = x[:,:,::-1]
    x = x / 255.0
    x = x[None,...]

    # Put in an input dict as per predict_fn input definition
    x = {'input': x}

    # Make predictions and fetch results from output dict
    predictions = predict_fn(x)
    x_src = predictions['x_src']
    x_gen = predictions['x_gen']

    # Show all source v.s. generated results
    compare_result(x_src, x_gen)
