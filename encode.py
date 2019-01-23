import numpy as np
import cv2
import tensorflow as tf
from tensorflow.contrib import predictor
import glob
from pathlib import Path
from sklearn import manifold
import imageio

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from config import *
from helper import *
from utils import *
from model_fn import *


tf.enable_eager_execution()

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        img = np.squeeze(imageData[i])
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

def make_gif():
    fl = glob.glob('./imgs/gif_img/*.png')
    list.sort(fl, key=lambda x: int(x.split('/')[-1].split('.png')[0]))
    images = []
    for f in fl:
        images.append(imageio.imread(f))

    imageio.mimsave('./imgs/gif.gif', images, fps=60)


def latent_interpolation(predict_fn):
    start_img  = load_img('./imgs/start.jpg')[None] / 255.0
    end_img = load_img('./imgs/end.jpg')[None] / 255.0

    x_val = np.vstack((start_img, end_img))

    dict_in = {'x': x_val, 'z': np.zeros(z_dim)[None]}
    start_mu, end_mu = predict_fn(dict_in)['mu']

    interp_steps = 300
    alpha_space = np.linspace(0, 1, interp_steps)
    interp_vals = []
    for alpha in alpha_space:
        val = start_mu*(1-alpha) + end_mu*alpha
        interp_vals.append(val)
    interp_vals = np.array(interp_vals)

    dict_in = {'x': np.zeros(input_dim)[None], 'z': interp_vals}
    interp_img = predict_fn(dict_in)['y']

    interp_img *= 255.0
    interp_img = interp_img.astype('uint8')
    #show_all(interp_img, interp_steps)

    for ii in range(interp_img.shape[0]):
        x = np.squeeze(interp_img[ii])
        x = x[:,:, ::-1]
        x = cv2.resize(x, (256, 256))
        cv2.imwrite('./imgs/gif_img/{}.png'.format(ii), x)

    make_gif()


def main(unused_argv):
    # Export model_fn to only use decoder
    export_tf_model(FLAGS.export_path)

    # Find latest frozen pb
    subdirs = [x for x in Path(FLAGS.export_path + '/frozen_pb').iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    # Create predictor
    predict_fn = predictor.from_saved_model(latest)

    dataset = tf.data.TFRecordDataset(FLAGS.test_path)\
              .repeat(1)\
              .map(extract_features, num_parallel_calls=4)\
              .batch(512)\
              .prefetch(8)

    iterator = dataset.make_one_shot_iterator()

    # Eager execution for obtaining batch data from dataset
    z_val = np.zeros(z_dim)[None]

    x_arr = np.zeros(input_dim)[None]
    mu_arr = np.zeros(z_dim)[None]

    latent_interpolation(predict_fn)

    ''' t-SNE visualization
    for ii, val in enumerate(iterator):
        x_val = val.numpy()
        dict_in = {'x': x_val, 'z': z_val}

        predictions = predict_fn(dict_in)
        x = predictions['x']
        mu = predictions['mu']

        x_arr = np.append(x_arr, x, axis=0)
        mu_arr = np.append(mu_arr, mu, axis=0)
        print('Finished Batch: {}'.format(ii))
        if ii == 10:
            break

    x_arr = x_arr[1:]
    mu_arr = mu_arr[1:]

    # t-SNE
    tsne = manifold.TSNE(n_components=2,
                         init='pca',
                         random_state=0,
                         learning_rate=80.0)

    X_tsne = tsne.fit_transform(mu_arr)

    print('Plotting t-SNE visualization...')
    fig, ax = plt.subplots()
    imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=x_arr, ax=ax, zoom=0.1)
    plt.show()
    '''


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('mode', None, 'TRAIN/TEST')
    tf.app.run()
