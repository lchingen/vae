import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(dataset_name='mnist'):
    dataset = getattr(tf.keras.datasets, dataset_name)
    (x_train, _), (x_test, _) = dataset.load_data()
    x_train, x_test = (x_train / 255.0), (x_test / 255.0)
    return x_train, x_test

def rmse(x, y):
    assert x.shape == y.shape
    N = x.size
    print('RMSE:{}'.format(np.sqrt(np.sum((x-y)**2) / N)))

def show_img(img, title=None):
    img = np.squeeze(img)
    plt.title(title)
    plt.imshow(img)
    plt.show()

def compare_result(old, new):
    new, old = np.squeeze(new), np.squeeze(old)
    f = plt.figure()

    f.add_subplot(1, 2, 1, title='Old')
    plt.imshow(old)

    f.add_subplot(1, 2, 2, title='New')
    plt.imshow(new)
    plt.show()

