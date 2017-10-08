from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
import tensorflow.contrib.slim as slim

from trainer.lenet import lenet

tf.logging.set_verbosity(tf.logging.INFO)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def input_fn(filename, batch_size=100):
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
        [image, label], batch_size=batch_size,
        capacity=1000 + 3 * batch_size)

    return {'inputs': images}, labels


def get_input_fn(filename, batch_size=100):
    return lambda: input_fn(filename, batch_size)


def _cnn_model_fn(features, labels, mode):
    # Input Layer
    inputs = tf.reshape(features['inputs'], [-1, 28, 28, 1])
    predictions = lenet(inputs)

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()

        one_hot_labels = slim.one_hot_encoding(labels, 10)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, predictions)
        tf.summary.scalar('Loss', loss)

        if mode == Modes.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        if mode == Modes.EVAL:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(tf.cast(labels, tf.int32),
                                                tf.argmax(input=predictions, axis=1))
            }
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=eval_metric_ops)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': tf.argmax(input=predictions, axis=1),
            'probabilities': slim.softmax(predictions),
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)


def build_estimator(model_dir):
    return tf.estimator.Estimator(
        model_fn=_cnn_model_fn,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
