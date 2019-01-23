import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


def rmse(x, y):
    assert x.shape == y.shape
    N = x.size
    print('RMSE:{}'.format(np.sqrt(np.sum((x-y)**2) / N)))


def load_dataset(dataset_name='mnist'):
    dataset = getattr(tf.keras.datasets, dataset_name)
    (x_train, _), (x_test, _) = dataset.load_data()
    x_train, x_test = (x_train / 255.0), (x_test / 255.0)
    return x_train, x_test


def show_img(img, title=None):
    img = np.squeeze(img)
    plt.title(title)
    plt.imshow(img)
    plt.show()


def compare(old, new):
    new, old = np.squeeze(new), np.squeeze(old)
    f = plt.figure()

    f.add_subplot(1, 2, 1, title='Old')
    plt.imshow(old)

    f.add_subplot(1, 2, 2, title='New')
    plt.imshow(new)
    plt.show()


def compare_all(x_org, x_gen, test_size):
    f = plt.figure()
    x_grid_size = int(np.sqrt(test_size))
    y_grid_size = int(np.sqrt(test_size))
    assert x_grid_size == y_grid_size # Just to make my life easy
    for ii in range(x_grid_size):
        for jj in range(y_grid_size):
            org = np.squeeze(x_org[x_grid_size*ii+jj])
            gen = np.squeeze(x_gen[x_grid_size*ii+jj])
            concat = np.hstack((org, gen))
            ax = f.add_subplot(y_grid_size, x_grid_size, ii*x_grid_size+jj+1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(concat)
    plt.tight_layout()
    plt.show()


def show_all(x, size):
    f = plt.figure()
    x_grid_size = 8
    y_grid_size = int(np.ceil(x_grid_size)) #TODO: BUF FIX size/...

    for ii in range(y_grid_size):
        for jj in range(x_grid_size):
            try:
                ax = f.add_subplot(y_grid_size, x_grid_size, ii*y_grid_size+jj+1)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(x[ii*y_grid_size+jj])
            except:
                break
    plt.tight_layout()
    plt.show()


def load_custom_dataset(dataset):
    path = './datasets/{}/{}.npy'.format(dataset, dataset)
    x = np.load(path)
    x = x / 255.0
    return x
