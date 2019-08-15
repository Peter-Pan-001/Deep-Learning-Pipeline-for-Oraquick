#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
sys.path.append("..")
from model import Model

class CNN(Model):
    """
    This is the CNN class representing a network for image classification. The
    network consists of convolutional layers, max-pooling layers, dense layers,
    and a dropout layer.

    @param filter_size (int): The initial filter size on the first layer, will be increased in every depth
    @param depth (int): Number of convolutions followed by pooling layers to be applied to the input
    @param net (dictionary): Holds convoluted/pooled tensor computations for quick access during debugging
    """
    def __init__(self, W, H, C, filter_size=16, depth=2, final_units=4096):
        self.img_width = W
        self.img_height = H
        self.img_channels = C
        self.initial_filter_size = filter_size
        self.depth = depth
        self.dense_units = final_units
        self.net = {}

    def json_serving_input_receiver_fn(self):
        """
        Function that accepts inference requests and prepares them for the model, and
        converts data from the input format into the feature tensors expected by the model.

        @return ServingInputReceiver
        """
        inputs = {
                'image': tf.placeholder(tf.float32, [None,
                                                     self.img_width,
                                                     self.img_height,
                                                     self.img_channels]),
        }

        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def build_architecture(self, X, mode):
        """
        Builds CNN architecture.

        @param X (Tensor/NumPy Array): Input tensor, generally in the [Wi, Hi, C] format
        @param mode (int): Mode indicating training or evaluation, recognized by TensorFlow
        @return feed (Tensor/NumPy Array): Output tensor, generally in the [Wo, Ho, C] format
        """
        X = X[list(X.keys())[0]]

        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        input_layer = tf.reshape(X, [-1, self.img_width, self.img_height, self.img_channels])

        feed = input_layer

        for i in range(self.depth):
            # Convolutional Layer
            # Computes  features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, IMG_WIDTH/(2**i), IMG_HEIGHT/(2**i), initial_filter_size*(i)]
            # Output Tensor Shape: [batch_size, IMG_WIDTH/(2**i), IMG_HEIGHT/(2**i), initial_filter_size*(i+1)]
            conv1 = tf.layers.conv2d(
                    inputs=feed,
                    filters=self.initial_filter_size*(i+1),
                    kernel_size=[3, 3],
                    padding="same",
                    data_format='channels_last',
                    activation=tf.nn.relu)

            # Pooling Layer
            # Max pooling with a 2x2 filter and stride of 2
            # Input Tensor Shape: [batch_size, IMG_WIDTH/(2**i), IMG_HEIGHT/(2**i), initial_filter_size*(i+1)]
            # Output Tensor Shape: [batch_size, IMG_WIDTH/(2**(i+1)), IMG_HEIGHT/(2**(i+1)), initial_filter_size*(i+1)]
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            feed = pool1 # update feed

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, IMAGE_SIZE/(2**depth), IMAGE_SIZE/(2**depth), initial_filter_size*(depth)]
        # Output Tensor Shape: [batch_size, IMAGE_SIZE/(2**depth) * IMAGE_SIZE/(2**depth) * initial_filter_size*(depth)]
        feed_flat = tf.reshape(feed,
                    [-1, int(self.img_width/(2**self.depth)) * int(self.img_height/(2**self.depth)) * self.initial_filter_size*(self.depth)])

        # Dense Layer, densely connected neurons
        # Input Tensor Shape: [batch_size, IMAGE_SIZE/(2**depth) * IMAGE_SIZE/(2**depth) * initial_filter_size*(depth)]
        # Output Tensor Shape: [batch_size, dense_units]
        dense = tf.layers.dense(inputs=feed_flat, units=self.dense_units, activation=tf.nn.relu)

        # Add dropout operation; 0.50 probability that element will be kept
        # We probably don't want to have a very high dropout rate
        # because the element may convey critical information
        # about the locations of a black dot on the test kit screen.
        dropout = tf.layers.dropout(inputs=dense, rate=0.50, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer = Output Layer
        # Input Tensor Shape: [batch_size, ?]
        # Output Tensor Shape: [batch_size, 3]
        # 3 is for the total number of possible cases of the ORAQUICK test kits.
        logits = tf.layers.dense(inputs=dropout, units=3)

        # 0 = Negative
        # 1 = positive
        # 2 = invalid

        # print(logits) # debugging
        return logits

    def generate_model(self, features, labels, mode):
        """
        Function to set prediction type and format, define the loss function for the model,
        set TensorFlow summary information, and configure evaluation options.

        @param features (Tensor): Input images for training, recognized by TensorFlow
        @param labels (Tensor): Ground-truth labels tensor for training, recognized by TensorFlow
        @param mode (int): Mode indicating training or evaluation, recognized by TensorFlow
        @return EstimatorSpec
        """
        output_classification = self.build_architecture(features, mode)

        predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=output_classification, axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": tf.nn.softmax(output_classification, name="softmax_tensor")
        }

        # Added export_outputs to save the model
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)})

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = output_classification,
                        labels = labels))

        # Visualize loss in TensorBoard to keep track of the training progress
        tf.summary.scalar('Loss', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Try different optimizers and learning rates
            # Ex: optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0010)
            # Learning rate could also be modified over time, exponential decay is preferred
            # Ex: rate = tf.train.exponential_decay(0.001, tf.train.get_global_step(), 1, 0.9999)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
                "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
                "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])
        }

        # Visualize accuracy, precision, and recall metrics in TensorBoard
        tf.summary.scalar('Accuracy', eval_metric_ops.get('accuracy'))
        tf.summary.scalar('Precision', eval_metric_ops.get('precision'))
        tf.summary.scalar('Recall', eval_metric_ops.get('recall'))

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
