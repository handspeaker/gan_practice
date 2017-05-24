# -*- coding: utf-8 -*-
import numpy as np
import os
import struct
import scipy.misc


def save_img(img, save_path, image_id):
    img = (img + 1.0) / 2.0
    scipy.misc.imsave(save_path + '/' + str(image_id) + '.jpg', img.reshape([img.shape[0], -1]))


def read_mnist_data(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        image = np.fromstring(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1).astype(np.float32)
        image = (image / 255.0 * 2 - 1)
    return image


class MnistProvider(object):
    def __init__(self, folder_path, shuffle=True):
        train_data_path = os.path.join(folder_path, 'train-images-idx3-ubyte')
        test_data_path = os.path.join(folder_path, 't10k-images-idx3-ubyte')

        self.train_data = read_mnist_data(train_data_path)
        self.val_data = read_mnist_data(test_data_path)
        if shuffle:
            perm = np.arange(self.train_data.shape[0])
            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]

        self.cursor = 0
        self.train_num = len(self.train_data)
        self.val_num = len(self.val_data)

    def next_train_batch(self, batch_size):
        if self.cursor + batch_size > self.train_num:
            perm = np.arange(self.train_num)
            np.random.shuffle(perm)
            self.train_data = self.train_data[perm]
            self.cursor = 0
        next_batch = self.train_data[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size
        return next_batch

    def get_val(self):
        return self.val_data

    def get_train_num(self):
        return self.train_num

    def get_val_num(self):
        return self.val_num
