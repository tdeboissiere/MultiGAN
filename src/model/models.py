import sys
import tensorflow as tf
sys.path.append("../utils")
import layers

FLAGS = tf.app.flags.FLAGS


class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class G16(Model):
    def __init__(self, name="G16"):

        super(G16, self).__init__(name)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            #################
            # Generator
            #################

            # Initial dense multiplication
            x = layers.linear(x, 512 * 8 * 8)

            # Reshape to image format
            if FLAGS.data_format == "NCHW":
                target_shape = (-1, 512, 8, 8)
            else:
                target_shape = (-1, 8, 8, 512)

            x = layers.reshape(x, target_shape)
            x = tf.contrib.layers.batch_norm(x, fused=True)
            x = tf.nn.elu(x)

            # Conv2D + Phase shift blocks
            x = layers.conv2d_block(x, "G16_conv2D_1", 256, 3, 1, data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "G16_conv2D_2", 256, 3, 1, data_format=FLAGS.data_format)
            x = layers.phase_shift(x, upsampling_factor=2, name="PS_G16", data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "G16_conv2D_3", FLAGS.channels, 3, 1, bn=False, activation_fn=None, data_format=FLAGS.data_format)
            x = tf.nn.tanh(x, name="x_G16")

            return x


class G32(Model):
    def __init__(self, name="G32"):

        super(G32, self).__init__(name)

    def __call__(self, x, x_feat, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            #################
            # Generator
            #################

            # Add x_feat
            up = x.get_shape().as_list()[2] / x_feat.get_shape().as_list()[2]
            x_feat = layers.phase_shift(x_feat, upsampling_factor=up, name="PS_G32_feat", data_format=FLAGS.data_format)
            if FLAGS.data_format == "NCHW":
                x = tf.concat([x, x_feat], axis=1)
            else:
                x = tf.concat([x, x_feat], axis=-1)

            x = layers.conv2d_block(x, "G32_conv2D_1", 256, 3, 1, data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "G32_conv2D_2", 256, 3, 1, data_format=FLAGS.data_format)
            x = layers.phase_shift(x, upsampling_factor=2, name="PS_G32", data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "G32_conv2D_3", FLAGS.channels, 3, 1, bn=False, activation_fn=None, data_format=FLAGS.data_format)
            x = tf.nn.tanh(x, name="x_G32")

            return x


class G64(Model):
    def __init__(self, name="G64"):

        super(G64, self).__init__(name)

    def __call__(self, x, x_feat, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            #################
            # Generator
            #################

            # Add x_feat
            up = x.get_shape().as_list()[2] / x_feat.get_shape().as_list()[2]
            x_feat = layers.phase_shift(x_feat, upsampling_factor=up, name="PS_G64_feat", data_format=FLAGS.data_format)
            if FLAGS.data_format == "NCHW":
                x = tf.concat([x, x_feat], axis=1)
            else:
                x = tf.concat([x, x_feat], axis=-1)

            x = layers.conv2d_block(x, "G64_conv2D_1", 256, 3, 1, data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "G64_conv2D_2", 256, 3, 1, data_format=FLAGS.data_format)
            x = layers.phase_shift(x, upsampling_factor=2, name="PS_G64", data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "G64_conv2D_3", FLAGS.channels, 3, 1, bn=False, activation_fn=None, data_format=FLAGS.data_format)
            x = tf.nn.tanh(x, name="x_G64")

            return x


class D16(Model):
    def __init__(self, name="D16"):
        # Determine data format from output shape

        super(D16, self).__init__(name)

    def __call__(self, x, reuse=False, mode="D"):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            x = layers.conv2d_block(x, "D16_conv2D_1", 32, 3, 2, data_format=FLAGS.data_format, bn=False)
            x = layers.conv2d_block(x, "D16_conv2D_2", 16, 3, 2, data_format=FLAGS.data_format)

            x_feat = tf.identity(x, "x_feat16")

            x_shape = x.get_shape().as_list()
            flat_dim = 1
            for d in x_shape[1:]:
                flat_dim *= d

            target_shape = (-1, flat_dim)
            x = layers.reshape(x, target_shape)

            x = layers.linear(x, 1)

            x_mbd = layers.mini_batch_disc(x, num_kernels=100, dim_per_kernel=5, name="mbd16")
            x = tf.concat([x, x_mbd], axis=1)

            if mode == "D":
                return x

            else:
                return x_feat, x


class D32(Model):
    def __init__(self, name="D32"):
        # Determine data format from output shape

        super(D32, self).__init__(name)

    def __call__(self, x, reuse=False, mode="D"):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            x = layers.conv2d_block(x, "D32_conv2D_1", 32, 3, 2, data_format=FLAGS.data_format, bn=False)
            x = layers.conv2d_block(x, "D32_conv2D_2", 64, 3, 2, data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "D32_conv2D_3", 64, 3, 2, data_format=FLAGS.data_format)

            x_feat = tf.identity(x, "x_feat32")

            x_shape = x.get_shape().as_list()
            flat_dim = 1
            for d in x_shape[1:]:
                flat_dim *= d

            target_shape = (-1, flat_dim)
            x = layers.reshape(x, target_shape)

            x_mbd = layers.mini_batch_disc(x, num_kernels=100, dim_per_kernel=5, name="mbd32")
            x = tf.concat([x, x_mbd], axis=1)

            x = layers.linear(x, 1)

            if mode == "D":
                return x

            else:
                return x_feat, x


class D64(Model):
    def __init__(self, name="D64"):
        # Determine data format from output shape

        super(D64, self).__init__(name)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()

            x = layers.conv2d_block(x, "D64_conv2D_1", 32, 3, 2, data_format=FLAGS.data_format, bn=False)
            x = layers.conv2d_block(x, "D64_conv2D_2", 64, 3, 2, data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "D64_conv2D_3", 128, 3, 2, data_format=FLAGS.data_format)
            x = layers.conv2d_block(x, "D64_conv2D_4", 256, 3, 2, data_format=FLAGS.data_format)

            x_shape = x.get_shape().as_list()
            flat_dim = 1
            for d in x_shape[1:]:
                flat_dim *= d

            target_shape = (-1, flat_dim)
            x = layers.reshape(x, target_shape)

            x_mbd = layers.mini_batch_disc(x, num_kernels=100, dim_per_kernel=5, name="mbd64")
            x = tf.concat([x, x_mbd], axis=1)

            x = layers.linear(x, 1)

            return x
