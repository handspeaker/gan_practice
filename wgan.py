# -*- coding: utf-8 -*-

# generate mnist style image by wgan

import os
import argparse
import tensorflow as tf
import numpy as np
import shutil
from dataset import MnistProvider, save_img
from ops import lrelu


def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32 = 128, 64, 32, 16  # channel num,256,128,64,32
    s4 = 4
    output_dim = 1  # gray image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # 4*4*256
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.layers.batch_normalization(conv1, training=is_train, name='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*64
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*256
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.layers.batch_normalization(conv4, training=is_train, name='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 32*32*1
        conv5 = tf.layers.conv2d_transpose(act4, output_dim, kernel_size=[32, 32], strides=[1, 1], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        act5 = tf.nn.tanh(conv5, name='act5')

        return act5


def discriminator(input, is_train, reuse=False):
    c2, c4, c8 = 16, 32, 64  # channel num,32, 64, 128
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        # 16*16*32
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        act1 = lrelu(conv1, n='act1')
        # 8*8*64
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.layers.batch_normalization(conv2, training=is_train, name='bn2')
        act2 = lrelu(bn2, n='act2')
        # 4*4*128
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[4, 4], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.layers.batch_normalization(conv3, training=is_train, name='bn3')
        act3 = lrelu(bn3, n='act3')

        shape = act3.get_shape().as_list()
        dim = shape[1] * shape[2] * shape[3]
        fc1 = tf.reshape(act3, shape=[-1, dim], name='fc1')
        w1 = tf.get_variable('w1', shape=[fc1.shape[1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        output = tf.add(tf.matmul(fc1, w1), b1, name='output')
        return output


def train(args):
    random_dim = args.rand_dim
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, [None, 32, 32, 1], name='mnist_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    fake_image = generator(random_input, random_dim, is_train)
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    # wgan loss
    d_loss = tf.reduce_mean(real_result - fake_result)  # This optimizes the discriminator.
    g_loss = tf.reduce_mean(fake_result)  # This optimizes the generator.

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    mnist_data = MnistProvider(args.data_path)
    epoch_num = args.epoch_num
    batch_size = args.batch_size
    batch_num = int(mnist_data.get_train_num() / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    print 'total training sample num:%d' % mnist_data.get_train_num()
    print 'batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, epoch_num)
    print 'start training...'
    for i in range(epoch_num):
        for j in range(batch_num):
            d_iters = 5
            # do more discriminator update at the begin
            if total_batch % 500 == 0 or total_batch < 25:
                d_iters = 25
            for k in range(d_iters):
                train_image = mnist_data.next_train_batch(batch_size)
                train_image = np.lib.pad(train_image, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                                         constant_values=(-1, -1))
                train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                sess.run(d_clip)
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            _, gLoss = sess.run([trainer_g, g_loss],
                                feed_dict={random_input: train_noise, is_train: True})

            print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
            total_batch += 1

        # test and save model every epoch, random select 100 samples from testset
        all_test_image = mnist_data.get_val()
        rand_arr = np.random.randint(0, mnist_data.get_val_num(), 100)
        test_image = all_test_image[rand_arr]
        test_image = np.lib.pad(test_image, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                                constant_values=(-1, -1))
        test_noise = np.random.uniform(-1.0, 1.0, size=[100, random_dim]).astype(np.float32)
        test_gLoss, test_dLoss, gen_images = sess.run([g_loss, d_loss, fake_image],
                                                      feed_dict={random_input: test_noise,
                                                                 real_image: test_image,
                                                                 is_train: True})
        # [gen_images] = sess.run([fake_image],feed_dict={random_input: test_noise,is_train: True})
        print 'epoch %d test: d_loss:%f,g_loss:%f' % (i, test_dLoss, test_gLoss)
        gen_images = np.asarray(gen_images, dtype=np.float32)
        curr_folder = os.path.join(args.model_dir, str(i))
        if os.path.exists(curr_folder):
            shutil.rmtree(curr_folder)
        os.mkdir(curr_folder)
        for m in range(gen_images.shape[0]):
            save_img(gen_images[m], curr_folder, m)
        # save check point every epoch
        saver.save(sess, args.model_dir + '/wgan_' + str(i) + '.cptk')


def infer(args):
    random_dim = args.rand_dim  # random input noise dimension
    with tf.variable_scope('input'):
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    fake_image = generator(random_input,random_dim, is_train)
    sess = tf.Session()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    model_path = tf.train.latest_checkpoint(args.model_dir, latest_filename=None)
    saver.restore(sess, model_path)
    test_noise = np.random.uniform(-1.0, 1.0, size=[10, random_dim]).astype(np.float32)
    [gen_images] = sess.run([fake_image], feed_dict={random_input: test_noise, is_train: True})
    gen_images = np.asarray(gen_images, dtype=np.float32)
    curr_folder = os.path.join(args.model_dir, 'infer')
    if os.path.exists(curr_folder):
        shutil.rmtree(curr_folder)
    os.mkdir(curr_folder)
    for m in range(gen_images.shape[0]):
        save_img(gen_images[m], curr_folder, m)
    print 'image generation success, check %s to see results' % curr_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data',
                        help='folder which stores mnist train and test files')
    parser.add_argument('--mode', type=str, default='train',
                        help='could be either infer or train')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='directory to save models')
    parser.add_argument('--batch_size', type=int, default='16',
                        help='train batch size')
    parser.add_argument('--epoch_num', type=int, default='10',
                        help='train epoch num')
    parser.add_argument('--rand_dim', type=int, default='128',
                        help='random input noise dimension')
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        infer(args)
    else:
        print "unknown mode!"
