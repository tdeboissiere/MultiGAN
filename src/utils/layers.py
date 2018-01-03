import numpy as np
import tensorflow as tf


def lrelu(x, leak=0.2):

    return tf.maximum(x, leak * x)


def reshape(x, target_shape):

    return tf.reshape(x, target_shape)


def linear(x, n_out, bias=False, name="linear"):

    with tf.variable_scope(name):

        n_in = x.shape[-1]

        # Initialize w
        w_init_std = np.sqrt(1.0 / n_out)
        w_init = tf.truncated_normal_initializer(0.0, w_init_std)
        w = tf.get_variable('w', shape=[n_in,n_out], initializer=w_init)

        # Dense mutliplication
        x = tf.matmul(x, w)

        if bias:

            # Initialize b
            b_init = tf.constant_initializer(0.0)
            b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

            # Add b
            x = x + b

        return x


def phase_shift(x, upsampling_factor=2, data_format="NCHW", name="PhaseShift"):

    if data_format == "NCHW":
        x = tf.transpose(x, [0,2,3,1])

    x = tf.depth_to_space(x, upsampling_factor, name=name)

    if data_format == "NCHW":
        x = tf.transpose(x, [0,3,1,2])

    return x


def mini_batch_disc(x, num_kernels=100, dim_per_kernel=5, name="mbd"):

    with tf.variable_scope(name):

        n_in = x.shape[-1]
        n_out = num_kernels * dim_per_kernel

        # Initialize w
        w_init_std = np.sqrt(1.0 / n_out)
        w_init = tf.truncated_normal_initializer(0.0, w_init_std)
        w = tf.get_variable('w', shape=[n_in,n_out], initializer=w_init)

        # Dense mutliplication
        x = tf.matmul(x, w)
        # Reshape x
        x = tf.reshape(x, [-1, num_kernels, dim_per_kernel])

        diffs = tf.expand_dims(x, 3) - tf.expand_dims(tf.transpose(x, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        x = tf.reduce_sum(tf.exp(-abs_diffs), 2)

        return x


def conv2d(x, n_in, n_out, k, s, p, bias=True, data_format="NCHW", name="conv2d", stddev=0.02):
    with tf.variable_scope(name):

        if data_format == "NHWC":
            strides = [1, s, s, 1]
        elif data_format == "NCHW":
            strides = [1, 1, s, s]

        # Initialize weigth
        w_init = tf.truncated_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k, k, n_in, n_out], initializer=w_init)

        # Compute conv
        conv = tf.nn.conv2d(x, w, strides=strides, padding=p, data_format=data_format)

        if bias:
            # Initialize bias
            b_init = tf.constant_initializer(0.0)
            b = tf.get_variable('b', shape=(n_out,), initializer=b_init)

            # Add bias
            conv = tf.reshape(tf.nn.bias_add(conv, b, data_format=data_format), conv.get_shape())

        return conv


def conv2d_block(x, name, f, k, s, p="SAME", stddev=0.02, data_format="NCHW", bias=False, bn=True, activation_fn=tf.nn.elu):
    with tf.variable_scope(name):

        if data_format == "NHWC":
            n_in = x.get_shape()[-1]
        elif data_format == "NCHW":
            n_in = x.get_shape()[1]

        n_out = f

        x = conv2d(x, n_in, n_out, k, s, p, stddev=stddev, name="conv2d", data_format=data_format, bias=bias)
        if bn:
            x = tf.contrib.layers.batch_norm(x, data_format=data_format, fused=True, scope="bn")
        if activation_fn is not None:
            x = activation_fn(x)

        return x
