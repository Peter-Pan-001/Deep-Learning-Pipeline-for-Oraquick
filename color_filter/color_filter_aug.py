import cv2
import numpy as np
import sys,os

IMG_WIDTH = 96
IMG_HEIGHT = 64
IMG_CHANNELS = 1

training_dir = 'crop_training'
testing_dir = 'crop_testing'
output_training_dir = 'color_filter_aug_output'
output_testing_dir = 'color_filter_aug_output'

"""
lower_red_1 = np.array([0,67,50])
upper_red_1 = np.array([10,255,255])

lower_red_2 = np.array([170,67,50])
upper_red_2 = np.array([180,255,255])
"""

lower_red_1 = np.array([0,70,50])
upper_red_1 = np.array([10,255,255])
lower_red_2 = np.array([170,70,50])
upper_red_2 = np.array([180,255,255])

# main function
for image_name in os.listdir(training_dir):
    if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
        img = cv2.imread('{}/{}'.format(training_dir, image_name))
        img = cv2.resize(img, (120, 64))
        img = img[:, 12:108]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = mask1 + mask2
        mask = np.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        for column in range(96):
                temp = mask[:, column][mask[:, column] >100]
                if len(temp) >= 5:   mask[:, column] = 255
                else: mask[:, column] = 0
        cv2.imwrite('./{}/{}'.format(output_training_dir, image_name),mask)

for image_name in os.listdir(testing_dir):
    if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
        img = cv2.imread('{}/{}'.format(testing_dir, image_name))
        img = cv2.resize(img, (120, 64))
        img = img[:, 12:108]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = mask1 + mask2
        mask = np.reshape(mask, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        for column in range(96):
                temp = mask[:, column][mask[:, column] >100]
                if len(temp) >= 8:   mask[:, column] = 255
                else: mask[:, column] = 0
        cv2.imwrite('./{}/{}'.format(output_testing_dir, image_name),mask)