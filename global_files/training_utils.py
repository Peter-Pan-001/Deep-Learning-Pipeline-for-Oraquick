#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday Feb 07 19:46:53 2019

@author: uzaymacar
Training Utilities
"""

import tensorflow as tf
import os
import cv2

def iou_loss(Y_true, Y_pred):
    """
    Calculates jaccardian index, also known as IOU, and returns it as a negative loss.
    @param Y_pred (Tensor), (N, H, W, 1): Prediction (logits) mask generated by the model
    @param Y_true (Tensor), (N, H, W, 1): Ground truth mask (label)
    @return 1 - IOU: Represents dissimilarity between tensors Y_pred and Y_true
    """
    # Flatten logits and labels to [BATCH SIZE, IMG_WIDTH*IMG_HEIGHT]
    Y_true = tf.keras.backend.flatten(Y_true)
    Y_pred = tf.keras.backend.flatten(Y_pred)

    # Calculate total pixel-wise intersection and union
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(Y_true * Y_pred))
    union = tf.keras.backend.sum(tf.keras.backend.abs(Y_true) + tf.keras.backend.abs(Y_pred))

    # Calculate Jaccard index
    numerator = intersection + tf.keras.backend.epsilon()
    denominator = union - intersection + tf.keras.backend.epsilon()
    jaccard_index = numerator / denominator

    return 1 - jaccard_index

def dice_loss(Y_true, Y_pred):
    """
    Calculates dice coefficient and returns it as a negative loss.
    @param Y_pred (Tensor), (N, H, W, 1): Prediction (logits) mask generated by the model
    @param Y_true (Tensor), (N, H, W, 1): Ground truth mask (label)
    @return 1 - Dice: Represents dissimilarity between tensors Y_pred and Y_true
    """
    # Flatten logits and labels to [BATCH SIZE, IMG_WIDTH*IMG_HEIGHT]
    Y_true = tf.keras.backend.flatten(Y_true)
    Y_pred = tf.keras.backend.flatten(Y_pred)

    # Calculate total pixel-wise intersection and union
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(Y_true * Y_pred))
    union = tf.keras.backend.sum(tf.keras.backend.abs(Y_true) + tf.keras.backend.abs(Y_pred))

    # Calculate Dice coefficient
    numerator = (2 * intersection) + tf.keras.backend.epsilon()
    denominator = union + tf.keras.backend.epsilon()
    dice_coefficient = numerator / denominator

    return 1 - dice_coefficient

def get_class_counts(filenames):
    """
    Function to return individual counts of classes for training/testing images
    according to their names.

    @param filenames (List): List to hold training/testing filenames
    @return Tuple: Specific counts for each classification class
    """

    invalid_count = 0 # number of invalid images in training/testing
    negative_count= 0 # number of negative images in training/testing
    positive_count = 0 # number of positive images in training/testing

    for filename in filenames:
        if "invalid" in filename.lower():
            invalid_count += 1
        elif "negative" in filename.lower():
            negative_count += 1
        elif "positive" in filename.lower():
            positive_count += 1
        else:
            print("Name convention not followed!")

    return (invalid_count, negative_count, positive_count)

def get_label(filename):
    """
    Function to correctly label files according to their names, and return
    the appropraite class representing the test case.

    @param filename (str): Filename string
    @return label (int): Specifying label for that class
    """

    if "invalid" in filename.lower():
        return 2 # Invalid Case
    elif "negative" in filename.lower():
        return 0 # Negative Case
    elif "positive" in filename.lower():
        return 1 # Positive Case
    else:
        print("Name convention not followed!")
        return None

def build_list_from_filepaths(filepaths):
    """
    Function to build list of images/data from given filepaths.

    @param filepaths (List): List containing relevant filepaths (full paths)
    @return data (List): List containing extracted images/data
    """
    data = []
    filenames = []
    for filename in os.listdir(filepaths):
         if "jpg" in filename or "jpeg" in filename or "png" in filename:
             # print(filename) # debugging
             #img = Image.open(filepaths + "/" + filename)
             img = cv2.imread(filepaths + "/" + filename)
             data.append(img)
             filenames.append(filename.split('.')[0])

    return data, filenames

def get_confidence_difference(class_confidences):
    """
    Function to estimate the overall confidence of the classification
    prediction by measuring the confidence difference between the
    most confident, predicted class, and the second, following next
    most confident class.

    @param class_confidences (List): (n*[P0, P1, P2, P3, P4]), list with confidences
           associated with each class
    @return confidence_difference
    """
    # Normalize the probabilities so they sum to 100 instead of 1
    class_confidences_normalized = class_confidences*100
    # Sort the normalized probabilities.
    sorted_confidences = sorted(class_confidences_normalized)

    # Get the most confident class that is predicted and returned, and the second most confident class
    most_confident_class = sorted_confidences[-1]
    second_most_confident_class = sorted_confidences[-2]

    # Compute the difference described in the docstring
    confidence_difference = most_confident_class - second_most_confident_class
    return confidence_difference