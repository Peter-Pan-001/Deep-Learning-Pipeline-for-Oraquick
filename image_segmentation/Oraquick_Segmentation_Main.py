#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard imports
import os
import cv2
import numpy as np
import tensorflow as tf

# Set path to parent directory for model and utilities
import sys
sys.path.append("..")

# Import UNET model
from segmentation_model import UNET

# Import utilities
from image_processing_utils import pairwise_augmentation, binary_to_color

# Specify parameters for images that will be fed in to the model
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

# Specify training paths for images and labels, and initialize lists
SEGMENTATION_TRAINING_IMAGES_PATH = "crop_training"
SEGMENTATION_TRAINING_LABELS_PATH = "crop_training_labels"
SEGMENTATION_TRAINING_IMAGES = []
SEGMENTATION_TRAINING_LABELS = []

# Specify testing paths for images and labels, and initialize lists
SEGMENTATION_TESTING_IMAGES_PATH = "crop_testing"
SEGMENTATION_TESTING_LABELS_PATH = "crop_testing_labels"
SEGMENTATION_TESTING_IMAGES = []
SEGMENTATION_TESTING_LABELS = []

# Specify export directory, and the model directory within this export directory
EXPORT_DIR = "saved_models/"
MODEL_DIR = "segmentation_model_001"
# Execute cd <EXPORT_DIR> & tensorboard --logdir=<MODEL_DIR> for TensorBoard visualization.
# Go to http://localhost:6006 in your browser to see training summary.

# Specify exactly one of number of training steps, or number of epochs
# An epoch is one iteration over all of the training data, whereas a step is one iteration over one training data
# TRAINING_STEPS = 10000
NUM_EPOCHS = 10 # selected number of epochs

# Batch size is the number of training examples utilized in one iteration
BATCH_SIZE = 8

# Specify the desired number of different variations (rotation, translation, scale) of each training image
AUGMENTATION_FACTOR = 16

# Initialize UNET model with image parameters
unet = UNET(IMG_WIDTH,
            IMG_HEIGHT,
            IMG_CHANNELS)

# tf.logging.set_verbosity(tf.logging.INFO) #TensorFlow will print all critical messages that have the label INFO.
# print(tf.VERSION) # check TensorFlow version

# 1) Setting up training lists for images and labels
for filename in os.listdir(SEGMENTATION_TRAINING_LABELS_PATH):
    if "jpg" in filename or "jpeg" in filename or "png" in filename: # check if image file
        # Labels are binary (grayscale) masks that are ground-truth, manually marked feature extractions
        label = cv2.imread(SEGMENTATION_TRAINING_LABELS_PATH + "/" + filename, 0) # mode = 0 for grayscale
        label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT)) # resize label
        label = np.reshape(label, (IMG_HEIGHT, IMG_WIDTH, 1)) # arrange dimensions
        # Images are RGB, contain the membrane of a test-kit (extracted with circle detection)
        img = cv2.imread(SEGMENTATION_TRAINING_IMAGES_PATH + "/" + filename, 1) # mode = 1 for RGB
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) # resize image

        # Perform augmentation
        augmented_imgs, augmented_labels = pairwise_augmentation(img, label, AUGMENTATION_FACTOR, filename)

        for aug_img in augmented_imgs:
            aug_img = aug_img / 255 # normalize augmented images for training
            SEGMENTATION_TRAINING_IMAGES.append(aug_img) # add to images list

        for aug_label in augmented_labels:
            # Apply threshold to binary augmented labels
            aug_label[aug_label<100] = 0
            aug_label[aug_label>=100] = 1
            SEGMENTATION_TRAINING_LABELS.append(aug_label) # add to labels list

        img = img / 255 # normalize image for training
        # Apply threshold to binary label
        label[label<100] = 0
        label[label>=100] = 1
        SEGMENTATION_TRAINING_IMAGES.append(img) # add to images list
        SEGMENTATION_TRAINING_LABELS.append(label) # add to labels list

# 2) Setting up testing lists for images and labels
testing_filenames = [] # will later use to display evaluation results correctly
for filename in os.listdir(SEGMENTATION_TESTING_LABELS_PATH):
    if "jpg" in filename or "jpeg" in filename or "png" in filename: # check if image file
        testing_filenames.append(filename) # add to filename list
        # Labels are binary (grayscale) masks that are ground-truth, manually marked feature extractions
        label = cv2.imread(SEGMENTATION_TESTING_LABELS_PATH + "/" + filename, 0) # mode = 0 for grayscale
        label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT)) # resize label
        label = np.reshape(label, (IMG_HEIGHT, IMG_WIDTH, 1)) # arrange dimensions
        # Images are RGB, contain the membrane of a test-kit (extracted with circle detection)
        img = cv2.imread(SEGMENTATION_TESTING_IMAGES_PATH + "/" + filename, 1) # mode = 1 for RGB
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) # resize image

        img = img / 255 # normalize image for testing
        # Apply threshold to binary label
        label[label<100] = 0
        label[label>=100] = 1
        SEGMENTATION_TESTING_IMAGES.append(img) # add to images list
        SEGMENTATION_TESTING_LABELS.append(label) # add to labels list

# 3) Set up training data and labels from training lists for images and labels
train_data = np.array(SEGMENTATION_TRAINING_IMAGES, dtype = np.float32)
train_labels = np.array(SEGMENTATION_TRAINING_LABELS, dtype = np.float32)

# 4) Set up evaluation data and labels from testing lists for images and labels
# NOTE: Evaluation and testing is used interchangeably here
# Change eval data and label to the training lists to see your training error
eval_data = np.array(SEGMENTATION_TESTING_IMAGES, dtype = np.float32)
eval_labels = np.array(SEGMENTATION_TESTING_LABELS, dtype = np.float32)

# 5) Create the Estimator
oraquick_segmentation_model = tf.estimator.Estimator(
        model_fn=unet.generate_model, # pass the final layer of the model
        model_dir=MODEL_DIR, # pass the pre-specified model directory
        config = tf.contrib.learn.RunConfig(
                save_checkpoints_steps=20,
                save_checkpoints_secs=None,
                save_summary_steps=40,))

# 6) Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities":"softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# 7) Train the model
# Create training input
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, # pass the training data
        y=train_labels, # pass the training labels
        batch_size=BATCH_SIZE, # pass the pre-specified batch size
        num_epochs=NUM_EPOCHS, # pass the pre-specified number of epochs
        shuffle=True) # TODO: Shuffle or not? Set to True by default

# Perform the actual training
oraquick_segmentation_model.train(
        input_fn=train_input_fn,
        #steps=TRAINING_STEPS, # uncomment to train with number of training steps instead of number of epochs
        hooks=[logging_hook])

# 8) Export the model
oraquick_segmentation_model.export_savedmodel(EXPORT_DIR, unet.json_serving_input_receiver_fn)

# 9) Evaluate the model
# Create evaluation input
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, # pass the evaluation data
        y=eval_labels, # pass the evaluation labels
        num_epochs=1, # pass number of epochs as 1 for evaluation
        shuffle=False) # no need to shuffle the evaluation data

# Perform the actual evaluation
eval_results = oraquick_segmentation_model.evaluate(input_fn=eval_input_fn)
# print(eval_results) # debugging

# 10) Predict the evaluation data with the trained model
# Don't forget to set yield_single_examples = False to prevent "IndexError: tuple index out of range"
test_results = oraquick_segmentation_model.predict(input_fn=eval_input_fn, yield_single_examples=False)

# 11) Visualize the predictions
counter = 0 # counter for filenames list
for i,p in enumerate(test_results):
    class_prediction = p.get('classes')
    for output in class_prediction: # for each prediction on each evaluation image
        output[output>=0.5] = 1 # set any pixel with probability >= 0.5 to 1
        output[output<0.5] = 0 # set any pixel with probability < 0.5 to 0
        show_mask = binary_to_color(output) # show_mask will show white pixels where output = 1
        cv2.imwrite('./output_images/{}'.format(testing_filenames[counter]), show_mask)
        counter += 1 # increment counter
