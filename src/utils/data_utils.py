import os
import sys
import glob
import tensorflow as tf
sys.path.append("../utils")
import logging_utils as lu


def normalize_image(image):

    image = tf.cast(image, tf.float32) / 255.
    image = (image - 0.5) / 0.5
    return image


def unnormalize_image(image, name=None):

    image = (image * 0.5 + 0.5) * 255.
    image = tf.cast(image, tf.uint8, name=name)
    return image


def read_celebA():

    FLAGS = tf.app.flags.FLAGS

    list_images = glob.glob(os.path.join(FLAGS.celebA_path, "*.jpg"))

    # Read each JPEG file
    with tf.device('/cpu:0'):

        reader = tf.WholeFileReader()
        filename_queue = tf.train.string_input_producer(list_images)
        key, value = reader.read(filename_queue)
        channels = FLAGS.channels
        image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
        image.set_shape([None, None, channels])

        # Center crop
        image = tf.image.central_crop(image, FLAGS.central_fraction)
        # Normalize
        image = normalize_image(image)

        # Resize
        image_16 = tf.image.resize_images(image, (16, 16), method=tf.image.ResizeMethod.AREA)
        image_32 = tf.image.resize_images(image, (32, 32), method=tf.image.ResizeMethod.AREA)
        image_64 = tf.image.resize_images(image, (64, 64), method=tf.image.ResizeMethod.AREA)

        # Format image to correct ordering
        if FLAGS.data_format == "NCHW":
            image_16 = tf.transpose(image_16, (2,0,1))
            image_32 = tf.transpose(image_32, (2,0,1))
            image_64 = tf.transpose(image_64, (2,0,1))

        # Using asynchronous queues
        img16_batch, img32_batch, img64_batch = tf.train.batch([image_16, image_32, image_64],
                                                               batch_size=FLAGS.batch_size,
                                                               num_threads=FLAGS.num_threads,
                                                               capacity=2 * FLAGS.num_threads * FLAGS.batch_size,
                                                               name='X_real_input')

        return img16_batch, img32_batch, img64_batch


def manage_queues(sess):

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    lu.print_queues()

    return coord, threads
