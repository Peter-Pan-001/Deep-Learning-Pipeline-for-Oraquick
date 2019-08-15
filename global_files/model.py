#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri March 30 15:31:23 2019

@author: uzaymacar

Model blueprint acting as superclass for specifically implemented models for
segmentation and classification. Merely acts a placeholder, and is included
for the purpose of generalization.
"""

class Model(object):
   """
   Model blueprint.

   Required Parameters:
   @param W (int): Width of the images that will be inputted
   @param H (int): Height of the images that will be inputted
   @param C (int): Number of channels of the images that will be inputted

   The values of W, H, C should be constant for an unique model, both for
   training and testing.
   """
   def __init__(self, W, H, C):
        self.img_width = W
        self.img_height = H
        self.img_channels = C

   def json_serving_input_receiver_fn(self):
       """
       Function that accepts inference requests and prepares them for the model, and
       converts data from the input format into the feature tensors expected by the model.

       @return ServingInputReceiver
       """
       pass

   def build_architecture(self):
       """
       Builds the model architecture.

       @return logits (Tensor/NumPy Array): Output tensor, generally in the [Wo, Ho, C] format
       """
       pass

   def generate_model(self):
       """
       Function to set prediction type and format, define the loss function for the model,
       set TensorFlow summary information, and configure evaluation options.

       @return EstimatorSpec
       """
       pass
