#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import sys
sys.path.append("..")
from model import Model

from training_utils import dice_loss

class UNET(Model):
    """
    This is the UNET class representing the architecture illustrated in the paper. The
    network consists of 3 parts: 1) Contracting Path, 2) Bottleneck,
    and 3) Expanding Path.

    @param features (List of integers): The feature values that will be used within each depth
         of the model. Conventionally, it is an ascending list.
    @param initial_filter_size (int): Represents the initial number of filters (features), which will
         double in consecutive depths. Original paper suggests 64.
    @param net (Dictionary): Will hold convoluted/pooled tensor computations for quick access during
         debugging.
    """
    def __init__(self, W, H, C, features=[32,64,128]):
        self.img_width = W
        self.img_height = H
        self.img_channels = C
        self.num_features = features
        self.net = {}

    def json_serving_input_receiver_fn(self):
        """
        Function that accepts inference requests and prepares them for the model, and
        converts data from the input format into the feature tensors expected by the model.

        @return ServingInputReceiver
        """
        inputs = {
                'image': tf.placeholder(tf.float32, [None,
                                                     self.img_height,
                                                     self.img_width,
                                                     self.img_channels]),
        }

        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def build_contracting_path(self, X, mode,  depth = 3):
        """
        Function for contracting/downsampling path which is formed with <depth> blocks each performing
        3 seperate layer operations: 1) 3x3 Convolution Layer, 2) 3x3 Convolution Layer,
        3) 2x2 Max Pooling

        @param X (Tensor/NumPy Array): Input tensor, generally in the [Wi, Hi, C] format
        @param mode (int): Mode indicating training or evaluation, recognized by TensorFlow
        @param depth (int): Depth of the contracting path, set to 3 by default
        @return feed (Tensor/NumPy Array): Output tensor, generally in the [Wo, Ho, C] format
        """
        feed = X

        for i in range(depth):
            # Iteration for block i + 1
            print(self.num_features[i])
            # Convolutional Layer 1
            # Computes (num_features[i]) features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv1 = tf.layers.conv2d(
                    inputs=feed,
                    filters=self.num_features[i],
                    kernel_size=[3, 3],
                    padding="same",
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.he_normal())

            # TODO: Will you add regularization (L1&L2) ?

            # Batch Normalization
            conv1_norm = tf.layers.batch_normalization(conv1)

            # Add to dictionary ('con' for contracting, 'conv' for convolution, 'B' for block)
            self.net['con_conv_B' + str(i + 1) + '_1'] = conv1_norm

            # Convolutional Layer 2
            # Computes (num_features[i]) features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv2 = tf.layers.conv2d(
                    inputs=conv1_norm,
                    filters=self.num_features[i],
                    kernel_size=[3, 3],
                    padding="same",
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.he_normal())

            # Batch Normalization
            conv2_norm = tf.layers.batch_normalization(conv2)
            # Add to dictionary ('con' for contracting, 'conv' for convolution, 'B' for block)
            self.net['con_conv_B' + str(i + 1) + '_2'] = conv2_norm

            # Pooling Layer 1
            # Max pooling layer with a 2x2 filter.
            pool1 = tf.layers.max_pooling2d(inputs=conv2_norm,
                                            pool_size=[2, 2],
                                            strides=2)
            # Add to dictionary ('con' for contracting, 'pool' for pool, 'B' for block)
            self.net['con_pool_B' + str(i + 1)] = pool1

            feed = pool1 # update feed

            if i >= depth - 2:
                # Dropout
                feed = tf.layers.dropout(
                        inputs = feed,
                        rate = 0.50,
                        training=mode == tf.estimator.ModeKeys.TRAIN)

        return feed

    def build_expanding_path(self, X, depth = 3):
        """
        Function for expanding/upsampling path which is formed with 4 blocks each performing
        4 seperate layer operations: 1) Deconvolution (Upsampling) Layer with stride = 2,
        2) Merge with corresponding feature map from contracting path,
        3) 3x3 Convolution Layer, 4) 3x3 Convolution Layer.

        @param X (Tensor/NumPy Array): Input tensor, generally in the [Wi, Hi, C] format
        @param depth (int): Depth of the expanding path, set to 3 by default
        @return feed (Tensor/NumPy Array): Output tensor, generally in the [Wo, Ho, C] format
        """
        feed = X
        # Loop for 4 blocks
        for i in range(depth):
            # Iteration for block i + 1
            print(self.num_features[::-1][i])
            # Deconvolution Layer 1 with stride = 2
            deconv1 = tf.layers.conv2d_transpose(
                    inputs=feed,
                    filters=self.num_features[::-1][i],
                    kernel_size=[2, 2],
                    strides = (2, 2),
                    padding="same",
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.he_normal())

            # Batch Normalization
            deconv1_norm = tf.layers.batch_normalization(deconv1)
            # Add to dictionary ('exp' for expanding, 'up' for upsampling, 'B' for block)
            self.net['exp_up_B' + str(i + 1)] = deconv1_norm

            # Merge with corresponding feature map from contracting path
            concat = tf.concat([deconv1_norm, self.net['con_conv_B' + str(depth-i) + '_2']], -1)
            # Add to dictionary ('exp' for expanding, 'con' for concatenation, 'B' for block)
            self.net['exp_concat_B' + str(i + 1)] = concat

            # Convolutional Layer 1
            # Computes (num_features[i]) features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv1 = tf.layers.conv2d(
                    inputs=deconv1_norm,
                    filters=self.num_features[::-1][i],
                    kernel_size=[3, 3],
                    padding="same",
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.he_normal())

            # Batch Normalization
            conv1_norm = tf.layers.batch_normalization(conv1)
            # Add to dictionary ('exp' for expanding, 'up' for upsampling, 'B' for block)
            self.net['exp_conv_B' + str(i + 1) + '_1'] = conv1_norm

            # Convolutional Layer 1
            # Computes (num_features[i]) features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            conv2 = tf.layers.conv2d(
                    inputs=conv1_norm,
                    filters=self.num_features[::-1][i],
                    kernel_size=[3, 3],
                    padding="same",
                    data_format='channels_last',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.he_normal())

            # Batch Normalization
            conv2_norm = tf.layers.batch_normalization(conv2)
            # Add to dictionary ('exp' for expanding, 'conv' for convolution, 'B' for block)
            self.net['exp_conv_B' + str(i + 1) + '_2'] = conv2_norm

            feed = conv2_norm # update feed

        return feed

    def build_architecture(self, X, mode):
        """
        Function to build the model architecture. Builds the contracting path,
        performs bottleneck, builds the expanding path, and creates an output
        segmentation as logits.

        @param X (Tensor/NumPy Array): Input tensor, generally in the [Wi, Hi, C] format
        @param mode (int): Mode indicating training or evaluation, recognized by TensorFlow
        @return output_segmentation (Tensor/NumPy Array): Output tensor, also known as logits
        """
        X = X[list(X.keys())[0]]

        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        input_layer = tf.reshape(X, [-1, self.img_height, self.img_width, self.img_channels])
        input_layer = tf.keras.layers.GaussianNoise(0.03)(input_layer)

        # Build contracting path
        con = self.build_contracting_path(input_layer, mode)

        # Bottleneck: 1) Convolutional Layer, 2) Convolutional Layer
        print(self.num_features[-1]*2)
        # Convolutional Layer 1
        conv1 = tf.layers.conv2d(
                inputs=con,
                filters=self.num_features[-1]*2,
                kernel_size=[3, 3],
                padding="same",
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.he_normal())

        # Batch Normalization
        conv1_norm = tf.layers.batch_normalization(conv1)
        # Add to dictionary ('bot' for bottleneck, 'conv' for convolution)
        self.net['bot_conv_1'] = conv1_norm

        # Convolutional Layer 2
        conv2 = tf.layers.conv2d(
                inputs=conv1_norm,
                filters=self.num_features[-1]*2,
                kernel_size=[3, 3],
                padding="same",
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.he_normal())

        # Batch Normalization
        conv2_norm = tf.layers.batch_normalization(conv2)
        # Add to dictionary ('bot' for bottleneck, 'conv' for convolution)
        self.net['bot_conv_2'] = conv2_norm

        # TODO: Are you going to enable dropout?
        # Dropout
        # drop = tf.layers.dropout(
                #inputs = conv2_norm,
                #rate = 0.50,
                #training=mode == tf.estimator.ModeKeys.TRAIN)

        # TODO: Are you going to add more droput layers or not?

        # Build expanding path
        exp = self.build_expanding_path(conv2_norm)

        exp_conv = tf.layers.conv2d(
                inputs=exp,
                filters=2,
                kernel_size=[3, 3],
                padding="same",
                data_format='channels_last',
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.he_normal())

        # Batch Normalization
        exp_conv_norm = tf.layers.batch_normalization(exp_conv)

        # Output Segmentation = Output Layer
        output_segmentation = tf.layers.conv2d(
                inputs=exp_conv_norm,
                filters=1,
                kernel_size=[1,1],
                data_format='channels_last',
                activation=tf.nn.sigmoid)

        # print(output_segmentation.shape) # debugging
        # print(output_segmentation) # debugging
        return output_segmentation

    def generate_model(self, features, labels, mode):
        """
        Function to set prediction type and format, define the loss function for the model,
        set TensorFlow summary information, and configure evaluation options.

        @param features (Tensor): Input images for training, recognized by TensorFlow
        @param labels (Tensor): Ground-truth labels tensor for training, recognized by TensorFlow
        @param mode (int): Mode indicating training or evaluation, recognized by TensorFlow
        @return EstimatorSpec
        """

        output_segmentation = self.build_architecture(features, mode)

        predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.identity(output_segmentation, name="output"),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
                "probabilities": tf.reduce_sum(
                        tf.reshape(output_segmentation, [-1, self.img_width*self.img_height]),
                        name="softmax_tensor")
        }

        # Added export_outputs to save the model
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)})

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = dice_loss(output_segmentation, labels)

        # Visualize logits and labels in TensorBoard to keep track of the training progress
        tf.summary.image('PREDICTION_MASK', output_segmentation)
        tf.summary.image('GROUND_TRUTH', labels)

        # Visualize loss in TensorBoard
        tf.summary.scalar('LOSS', loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Try different optimizers and learning rates
            # Ex: optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0010)
            # Learning rate could also be modified over time, exponential decay is preferred
            # Ex: rate = tf.train.exponential_decay(0.001, tf.train.get_global_step(), 1, 0.9999)
            initial_learning_rate = 1E-4
            rate = tf.train.exponential_decay(initial_learning_rate, tf.train.get_global_step(), 500, 0.97)
            optimizer = tf.train.AdamOptimizer(learning_rate = rate)
            train_op = optimizer.minimize(
                        loss=loss,
                        global_step=tf.train.get_global_step())

            # Visualize learning rate and its decay in TensorBoard
            tf.summary.scalar('LEARNING_RATE', rate)

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
