import os
import json
from bunch import Bunch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def data_loader(config):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    x = x.astype('float32') / 255.0
    y = np.concatenate((y_train, y_test))
    trainloader = tf.data.Dataset.from_tensor_slices((x, y)).batch(config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return x, y, trainloader

def save_images(images, path):
    num_samples, h, w, c = images.shape[0], images.shape[1], images.shape[2], images.shape[3]
    frame_dim = int(np.sqrt(num_samples))
    canvas = np.squeeze(np.zeros((h * frame_dim, w * frame_dim, c)))
    for idx, image in enumerate(images):
        i = idx // frame_dim
        j = idx % frame_dim
        if c==1:
            canvas[i*h : (i+1)*h, j*w : (j+1)*w] = np.squeeze(image)
        elif c==3:
            canvas[i*h : (i+1)*h, j*w : (j+1)*w, :] = image
        else:
            print('Image channels must be 1 or 3!')
    if c==1:
        plt.imsave(path, canvas, cmap='gray')
    if c==3:
        plt.imsave(path, canvas)

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    return config

def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
