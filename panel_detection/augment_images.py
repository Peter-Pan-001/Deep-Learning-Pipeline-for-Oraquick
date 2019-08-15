#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 06:07:57 2019

@author: uzaymacar

Script for performing data augmentation on images through various, randomly
occurring image processing methods:
1) horizontal flip, 2) random crop, 3) affine transformation,
4) averaging over superpixels, 5) blurring, 6) sharpening,
7) edge detection and marking, 8) Gaussian noise, 9) pixel dropout,
10) pixel inversion, 11) pixel addition, 12) brightness modification,
13) contrast modification, 14) grayscale conversion, 15) elastic transformation,
and 16) image distortion.

It should be noted that through imgaug's iaa.Sometimes, we apply the above
methods in randomized order to randomly picked images, to have a diversity
within and across each batch.

The sequential image augmentation steps and methodologies are taken directly from
https://imgaug.readthedocs.io/en/latest/source/examples_basics.html. More could
be found in the documentations of imgaug.

Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --input_dir=training_images --output_dir=augmented_images
  # Create test data:
  python generate_tfrecord.py --input_dir=training_images --output_dir=augmented_images

  # output_dir can be the same as input_dir to save augmented images on the same directory.
"""

# Standard imports
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import tensorflow as tf

# Utility import
import sys
sys.path.append('/Users/panzichen/Image_Processing_Mine')
from training_utils import build_list_from_filepaths

# Define flags that will be inputted by the user
flags = tf.app.flags
flags.DEFINE_string('input_dir', '', 'Path to the directory that contains input images')
flags.DEFINE_string('output_dir', '', 'Path to the directory where augmented images will be saved to')
FLAGS = flags.FLAGS

# Set seed parameters for imgaug and define random probability function with p=0.5
ia.seed(5)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Construct a sequence of image augmentation methods
seq = iaa.Sequential(
    [
        # Apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images

        # Crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )),


        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        iaa.SomeOf((0, 2),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's chanel with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # Do all of the above augmentations in random order
            random_order=True
        )
    ],
    # Do all of the above augmentations in random order
    random_order=True
)

def batch_augment(images, filenames, num_iterations=1):
    """
    Function to perform batch augmentation through the imgaug sequence defined
    above, and save to the desired output directory.

    @param images (List/NumPy array): List consisting of input images. Should be either a 4D
         numpy array of shape (N, height, width, channels) or a list of 3D numpy arrays,
         each having shape (height, width, channels)
    @param num_iterations (int): Number of iterations over input images, set to once by default
    @return None
    """
    for batch_idx in range(num_iterations):
        # 'images'
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in
        # range 0-255.
        images_aug = seq.augment_images(images)
        i = 0
        for img in images_aug:
            cv2.imwrite(str(str(FLAGS.output_dir) + '/{}_augimg{}.jpg').format(filenames[i], batch_idx+1), img)
            i+=1

def main(_):
    """
    Main function that will be executed with parameters as those
    specified in FLAGS. Creates an input list and calls augmentation.
    """
    # Construct list containing input images
    images, filenames = build_list_from_filepaths(FLAGS.input_dir)
    # Perform augmentation
    batch_augment(images, filenames, 5)

if __name__ == '__main__':
    tf.app.run()
