# -*- coding: utf-8 -*-
import numpy as np
import os
import struct
import scipy.misc


def render_fonts_image(x, path, img_per_row, unit_scale=True):
    if unit_scale:
        # scale 0-1 matrix back to gray scale bitmaps
        bitmaps = (x * 255.).astype(dtype=np.int16) % 256
    else:
        bitmaps = x
    num_imgs, h, w = x.shape
    width = img_per_row * w
    height = int(np.ceil(float(num_imgs) / img_per_row)) * h
    canvas = np.zeros(shape=(height, width), dtype=np.int16)
    # make the canvas all white
    canvas.fill(0)
    for idx, bm in enumerate(bitmaps):
        x = h * int(idx / img_per_row)
        y = w * int(idx % img_per_row)
        canvas[x: x + h, y: y + w] = bm
    scipy.misc.toimage(canvas).save(path)
    return path


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
