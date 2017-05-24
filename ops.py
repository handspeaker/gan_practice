# -*- coding: utf-8 -*-
import tensorflow as tf


def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)
