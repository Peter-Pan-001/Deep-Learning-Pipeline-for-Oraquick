import cv2
import numpy as np
import sys,os

IMG_WIDTH = 96
IMG_HEIGHT = 64
IMG_CHANNELS = 1

training_dir = 'classification_training_original'
testing_dir = 'classification_testing_original'
output_training_dir = 'flip_training'
output_testing_dir = 'flip_testing'

# main function
for image_name in os.listdir(training_dir):
    if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
        img = cv2.imread('{}/{}'.format(training_dir, image_name))
        img1 = cv2.flip(img, 0)
        img2 = cv2.flip(img, 1)
        img3 = cv2.flip(img, -1)
        cv2.imwrite('./{}/{}.jpg'.format(output_training_dir, image_name.split('.')[0] + 'flip1'),img1)
        cv2.imwrite('./{}/{}.jpg'.format(output_training_dir, image_name.split('.')[0] + 'flip2'),img2)
        cv2.imwrite('./{}/{}.jpg'.format(output_training_dir, image_name.split('.')[0] + 'flip3'),img3)

for image_name in os.listdir(testing_dir):
    if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
        img = cv2.imread('{}/{}'.format(testing_dir, image_name))
        img1 = cv2.flip(img, 0)
        img2 = cv2.flip(img, 1)
        img3 = cv2.flip(img, -1)
        cv2.imwrite('./{}/{}.jpg'.format(output_testing_dir, image_name.split('.')[0] + 'flip1'),img1)
        cv2.imwrite('./{}/{}.jpg'.format(output_testing_dir, image_name.split('.')[0] + 'flip2'),img2)
        cv2.imwrite('./{}/{}.jpg'.format(output_testing_dir, image_name.split('.')[0] + 'flip3'),img3)