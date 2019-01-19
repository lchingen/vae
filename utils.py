import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

from config import *


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_db_sets(path, shuffle=True, split = [0.85, 0.14, 0.01]):
    file_paths = glob.glob('{}/*.jpg'.format(path))+\
                 glob.glob('{}/*.png'.format(path))

    if shuffle:
        np.random.shuffle(file_paths)

    num_samples = len(file_paths)
    train_ratio, val_ratio, test_ratio = split
    a = int(train_ratio*num_samples)
    b = a + int(val_ratio*num_samples)

    train = file_paths[:a]
    val   = file_paths[a:b]
    test  = file_paths[b:]
    return train, val, test


def load_img(path, size=(input_dim[0], input_dim[1]), data_type=np.float32):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(data_type)
    return img


def create_tf_record(path_set ,set_name):
    record_path = './db/{}.tfrecords'.format(set_name)
    writer = tf.python_io.TFRecordWriter(record_path)

    for path in path_set:
        img = load_img(path)
        feature = {'img':_bytes_feature(tf.compat.as_bytes(img.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def extract_features(example):
    feature = {'img': tf.FixedLenFeature([], tf.string)}
    parsed_example = tf.parse_single_example(example, feature)
    imgs = tf.decode_raw(parsed_example['img'], tf.float32)
    imgs = tf.reshape(imgs, input_dim)
    imgs /= 255.0
    return imgs


def augment_features(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=32.0/255.0)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def create_dataset(path, buffer_size, batch_size, num_epochs):
    # NOTE: change the extract_feature reshape size for different datasets
    with tf.device('cpu:0'):
        dataset = tf.data.TFRecordDataset('./db/train.tfrecords')\
                  .shuffle(buffer_size)\
                  .repeat(num_epochs)\
                  .map(extract_features, num_parallel_calls=4)\
                  .map(augment_features, num_parallel_calls=4)\
                  .batch(batch_size)\
                  .prefetch(1)
        return dataset
