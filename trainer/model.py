from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes

from trainer.lenet import LeNet

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
    input_layer = tf.reshape(features['inputs'], [-1, 28, 28, 1])

    net = LeNet({'data': input_layer})
    outputs = net.get_output()

    # Define operations
    if mode in (Modes.PREDICT, Modes.EVAL):
        # predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = outputs
        predicted_indices = tf.argmax(input=outputs, axis=1)

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(label_indices, depth=10), logits=outputs)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def build_estimator(model_dir):
    return tf.estimator.Estimator(
        model_fn=_cnn_model_fn,
        model_dir=model_dir,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)
