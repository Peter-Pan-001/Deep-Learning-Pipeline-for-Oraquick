#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday Feb 07 19:46:53 2019

@author: uzaymacar
Image Processing Utilities
"""

import numpy as np
import cv2
import random
import tensorflow as tf
import scipy
import os

def binary_to_color(mask, thresholded = True, threshold = 127):
    """
    Converts binary mask [0-1] to color (RGB or grayscale) [0-255]
    @param mask (ndarray): Binary mask
    @param thresholded (boolean): Set true to round pixels to edge intensities 0 and 255
    @param threshold (int): Pixels divided according to this seperator value
    @return color_mask (ndarray): Color mask
    """
    color_mask = mask * 255
    if thresholded:
        color_mask[color_mask>=threshold] = 255
        color_mask[color_mask<threshold] = 0
    return color_mask

def color_to_binary(mask, thresholded = True, threshold = 0.5):
    """
    Converts color mask (RGB or grayscale) [0-255] to binary [0-1].
    @param mask (ndarray): Color mask
    @param thresholded (boolean): Set true to round pixels to edge intensities 0 and 1
    @param threshold (int): Pixels divided according to this seperator value
    @return bin_mask (ndarray): Binary mask
    """
    bin_mask = mask / 255
    if thresholded:
        bin_mask[bin_mask>=threshold] = 1
        bin_mask[bin_mask<threshold] = 0
    return bin_mask

def pixel_wise_softmax(output_map):
    """
    Converts an un-normalized tensor to a normalized tensor through the
    probability distribution by softmax function.
    @param output_map (Tensor): Output segmentation map produced by model
    @param normalized_map (Tensor): Normalized segmentation map
    """
    # Calculate maximum pixel intensity
    max_pixel_intensity = tf.reduce_max(output_map, axis=3, keepdims=True)

    # Calculate element-wise exponential of output map, numerator for softmax function
    exponential_map = tf.exp(output_map - max_pixel_intensity)
    # Calculate sum of element-wise exponentials of output map, denominator for softmax function
    sum_of_exponentials = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
    # Calculate softmax
    normalized_map = exponential_map / sum_of_exponentials

    return normalized_map

def pairwise_augmentation(img, label, num_out, filename):
    """
    Adapted from https://github.com/spongezhang/iphone_image_segmentation/blob/master/code/one_shot_training_sequence.py
    Function to generate new sample image by performing image augmentation with
    1) Rotation: Rotates images and masks by random angles,
    2) Shearing: Creates realistic shape distortions using a shearing matrix,
    3) Translation: Horizontal and vertcical shift using a translation matrix.

    @param image (3D ndarray) (H, W, D): Training image
    @param label (1D ndarray): Binary mask representing the training image
    @param num_out (int): Augmentation factor, number of augmented images desired
    @return Tuple(augmented_imgs (List), augmented_labels (List)): Augmented training samples
    """
    augmented_imgs = []
    augmented_labels = []

    for i in range(num_out):

        # Select augmentation parameters
        angle = random.uniform(-10, 10) # random angle in the range [-10, 10]
        scale = random.uniform(0.95, 1.05) # random scaling factor in the range [0.95, 1.05]

        W, H = img.shape[1], img.shape[0]

        tmp_rotation_matrix = cv2.getRotationMatrix2D((W/2, H/2),
                                                      angle=angle,
                                                      scale=scale)
        rotation_matrix = np.eye(3, dtype=np.float32)
        rotation_matrix[0:2, :] = tmp_rotation_matrix

        shearing_matrix = np.eye(3, dtype=np.float32)
        shearing_matrix[0,1] = random.uniform(-0.005, 0.005)
        shearing_matrix[1,0] = random.uniform(-0.005, 0.005)

        translation_matrix = np.eye(3, dtype=np.float32)
        translation_matrix[0,2] = random.randint(-10, 10)
        translation_matrix[1,2] = random.randint(-10, 10)

        transform_matrix = np.matmul(translation_matrix, np.matmul(shearing_matrix, rotation_matrix))

        transformed_image = cv2.warpPerspective(img, transform_matrix,
                                                (W, H),
                                                flags=cv2.INTER_LINEAR,
                                                borderValue=(255,255,255))

        transformed_mask = np.zeros((H, W), dtype = np.uint8)

        temp_mask = cv2.warpPerspective(label, transform_matrix,
                                        (W, H),
                                        flags=cv2.INTER_NEAREST,
                                        borderValue=(0))

        transformed_mask[temp_mask>100] = 255

        augmented_image = transformed_image

        
        cv2.imwrite('/Users/panzichen/Image_Processing_Mine/AUGMENTATION/{}_image.jpg'.format(filename + str(i)), augmented_image)
        cv2.imwrite('/Users/panzichen/Image_Processing_Mine/AUGMENTATION/{}_mask.jpg'.format(filename + str(i)), transformed_mask)

        transformed_mask = np.reshape(transformed_mask, (W, H, 1))
        transformed_mask = color_to_binary(transformed_mask, threshold=100)

        augmented_image = color_to_binary(augmented_image, thresholded=False)

        augmented_imgs.append(augmented_image)
        augmented_labels.append(transformed_mask)

    return augmented_imgs, augmented_labels