#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard imports
import os
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("..")
from classification_model import CNN
from training_utils import build_list_from_filepaths, get_label, get_class_counts, get_confidence_difference

# Specify parameters for images that will be fed in to the model
IMG_WIDTH = 96
IMG_HEIGHT = 64
IMG_CHANNELS = 1

# Specify training paths for images and labels, and initialize lists
CLASSIFICATION_TRAINING_IMAGES_PATH = "classification_training"
CLASSIFICATION_TRAINING_IMAGES = []
CLASSIFICATION_TRAINING_LABELS = []

# Specify testing paths for images and labels, and initialize lists
CLASSIFICATION_TESTING_IMAGES_PATH = "classification_testing"
CLASSIFICATION_TESTING_IMAGES = []
CLASSIFICATION_TESTING_LABELS = []

# Specify export directory, and the model directory within this export directory
EXPORT_DIR = "saved_models/"
MODEL_DIR = "classification_model"
# Execute cd <EXPORT_DIR> & tensorboard --logdir=<MODEL_DIR> for TensorBoard visualization.
# Go to http://localhost:6006 in your browser to see training summary.

# Specify exactly one of number of training steps, or number of epochs
# An epoch is one iteration over all of the training data, whereas a step is one iteration over one training data
# TRAINING_STEPS = 10000
NUM_EPOCHS = 10 # selected number of epochs

# Batch size is the number of training examples utilized in one iteration
BATCH_SIZE = 16

# Copying factors to change when applying data augmentation to data
TRAINING_FACTOR = 1
TESTING_FACTOR = 1

# Initialize CNN model with image parameters
cnn = CNN(IMG_WIDTH,
          IMG_HEIGHT,
          IMG_CHANNELS)

# tf.logging.set_verbosity(tf.logging.INFO) #TensorFlow will print all critical messages that have the label INFO.
# print(tf.VERSION) # check TensorFlow version

# 1) Setting up training lists for images and labels
raw_training_images, _ = build_list_from_filepaths(CLASSIFICATION_TRAINING_IMAGES_PATH)
for img in raw_training_images:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
        CLASSIFICATION_TRAINING_IMAGES.append(img)
CLASSIFICATION_TRAINING_IMAGES = np.array(CLASSIFICATION_TRAINING_IMAGES, dtype=np.float32)
num_training_images = (int) (CLASSIFICATION_TRAINING_IMAGES.size/(IMG_WIDTH*IMG_HEIGHT)) # get number of training images

# Create labels from filenames
for filename in os.listdir(CLASSIFICATION_TRAINING_IMAGES_PATH):
    if "jpg" in filename or "jpeg" in filename or "png" in filename:
        # print("train: " + filename) # debugging
        for i in range(TRAINING_FACTOR):
            CLASSIFICATION_TRAINING_LABELS.append(get_label(filename))


# 2) Setting up testing lists for images and labels
raw_testing_images, _ = build_list_from_filepaths(CLASSIFICATION_TESTING_IMAGES_PATH)
for img in raw_testing_images:
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
        CLASSIFICATION_TESTING_IMAGES.append(img)
CLASSIFICATION_TESTING_IMAGES = np.array(CLASSIFICATION_TESTING_IMAGES, dtype=np.float32)
num_testing_images = (int) (CLASSIFICATION_TESTING_IMAGES.size/(IMG_WIDTH*IMG_HEIGHT)) # get number of testing images

# Create labels from filenames
testing_filenames = [] # will later use to display evaluation results correctly
for filename in os.listdir(CLASSIFICATION_TESTING_IMAGES_PATH):
    if "jpg" in filename or "jpeg" in filename or "png" in filename:
        # print("test: " + filename) # debugging
        testing_filenames.append(filename)
        for i in range(TESTING_FACTOR):
            CLASSIFICATION_TESTING_LABELS.append(get_label(filename))

# 3) OPTIONAL: Check class counts for both training and testing data
# Depending on application, you usually need relatively high number of examples for each class
# in order to prevent underfitting. Often, a balanced class count will prevent overfitting.
print("TRAINING COUNTS: [INVALID], [NEGATIVE], [POSITIVE]")
print(get_class_counts(os.listdir(CLASSIFICATION_TRAINING_IMAGES_PATH)))
print("TESTING COUNTS: [INVALID], [NEGATIVE], [POSITIVE]")
print(get_class_counts(os.listdir(CLASSIFICATION_TESTING_IMAGES_PATH)))

# 4) Set up training data and labels from training lists for images and labels
train_data = CLASSIFICATION_TRAINING_IMAGES
train_labels = np.array(CLASSIFICATION_TRAINING_LABELS, dtype = np.int32)

# 5) Set up evaluation data and labels from testing lists for images and labels
# NOTE: Evaluation and testing is used interchangeably here
# Change eval data and label to the training lists to see your training error
eval_data = CLASSIFICATION_TESTING_IMAGES
eval_labels = np.array(CLASSIFICATION_TESTING_LABELS, dtype = np.int32)

# 6) Create the Estimator
insti_classification_model = tf.estimator.Estimator(
        model_fn=cnn.generate_model, # pass the final layer of the model
        model_dir=MODEL_DIR, # pass the pre-specified model directory
        config = tf.contrib.learn.RunConfig(
                save_checkpoints_steps=20,
                save_checkpoints_secs=None,
                save_summary_steps=40,))

# 7) Set up logging for predictions
# Log the values in the "Softmax" tensor with label "probabilities"
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# 8) Train the model
# Create training input
train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, # pass the training data
        y=train_labels, # pass the training labels
        batch_size=BATCH_SIZE, # pass the pre-specified batch size
        num_epochs=NUM_EPOCHS, # pass the pre-specified number of epochs
        shuffle=True) # TODO: Shuffle or not? Set to True by default

# Perform the actual training
insti_classification_model.train(
      input_fn=train_input_fn,
      #steps=TRAINING_STEPS, # uncomment to train with number of training steps instead of number of epochs
      hooks=[logging_hook])

# 9) Export the model
insti_classification_model.export_savedmodel(EXPORT_DIR, cnn.json_serving_input_receiver_fn)

# 10) Evaluate the model
# Create evaluation input
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, # pass the evaluation data
        y=eval_labels, # pass the evaluation labels
        num_epochs=1, # pass number of epochs as 1 for evaluation
        shuffle=False) # no need to shuffle the evaluation data

# Perform the actual evaluation
eval_results = insti_classification_model.evaluate(input_fn=eval_input_fn)
# print(eval_results) # debugging

# 11) Predict the evaluation data with the trained model
test_results = insti_classification_model.predict(input_fn=eval_input_fn)

# 12) Visualize, analyze, and quantify the predictions
prediction_labels = [] # prediction labels
prediction_confidences = [] # confidences of each individual label for the prediction, in percentage
prediction_confidences_differences = [] # defined as the percentage difference between the first and second most confident labels

for i,p in enumerate(test_results):
    # Get the class prediction; either 0 (negative), 1 (positive), 2 (invalid)
    class_prediction = p.get('classes')
    prediction_labels.append(int(class_prediction))

    # Get the class confidences as an array.
    class_confidences = p.get('probabilities')
    prediction_confidences.append(class_confidences)

    # Get the confidence difference the most confident predicted class and the second most confident class
    confidence_difference = get_confidence_difference(class_confidences)
    prediction_confidences_differences.append(int(confidence_difference))

    # Print the class prediction and confidence difference for debugging and evaluation
    print(str(testing_filenames[int(i/TESTING_FACTOR)]),
          str(class_prediction),
          str(confidence_difference))

correct_prediction_count = 0 # count total number of correct predictions
for i in range(num_testing_images):
    if prediction_labels[i] == CLASSIFICATION_TESTING_LABELS[i]: # compare predictions to ground-truth
        correct_prediction_count += 1 # update correct prediction counter
        print(testing_filenames[i] + ": ",
              str(prediction_labels[i]),
              "; CORRECT")
    else: # if here, predicted class doesn't agree with the ground-truth class
        print(testing_filenames[i] + ": ",
              "GUESSED " + str(prediction_labels[i]) + " BUT IT WAS " + str(CLASSIFICATION_TESTING_LABELS[i]),
              "; WRONG!")

# Print final evaluation results
print("CORRECTLY GUESSED: " + str(correct_prediction_count) + " OUT OF " + str(num_testing_images))
print("ACCURACY: %" + str(float(correct_prediction_count/num_testing_images)*100))
