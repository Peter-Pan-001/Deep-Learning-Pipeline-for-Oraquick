import cv2
import numpy as np
import sys,os
import imutils
import xml.etree.ElementTree as ET
from xml.dom import minidom

# This script is to rotate the pictures clockwise and in the meantime adjust bounding label xml

input_training_dir = 'original_training_images'
input_training_xml_dir = 'original_training_labels_xml'
input_testing_dir = 'original_testing_images'
input_testing_xml_dir = 'original_testing_labels_xml'
output_training_dir = 'new_training_images'
output_training_xml_dir = 'new_training_labels_xml'
output_testing_dir = 'new_testing_images'
output_testing_xml_dir = 'new_testing_labels_xml'

for image_name in os.listdir(input_training_dir):
    if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
        img = cv2.imread('{}/{}'.format(input_training_dir, image_name))
        img1 = imutils.rotate_bound(img, 90)
        cv2.imwrite('./{}/{}.jpg'.format(output_training_dir, image_name.split('.')[0] + '_rotate'),img1)

        # revise corresponding xml
        for xml_name in os.listdir(input_training_xml_dir):
                if 'xml' in xml_name and xml_name.split('.')[0] == image_name.split('.')[0]:

                        # read xml
                        xml_doc = minidom.parse('{}/{}'.format(input_training_xml_dir, xml_name))
                        shape1 = int(xml_doc.getElementsByTagName('width')[0].childNodes[0].data)
                        shape0 = int(xml_doc.getElementsByTagName('height')[0].childNodes[0].data)
                        xmin = int(xml_doc.getElementsByTagName('xmin')[0].childNodes[0].data)
                        xmax = int(xml_doc.getElementsByTagName('xmax')[0].childNodes[0].data)
                        ymin = int(xml_doc.getElementsByTagName('ymin')[0].childNodes[0].data)
                        ymax = int(xml_doc.getElementsByTagName('ymax')[0].childNodes[0].data)

                        # change xml
                        tree = ET.parse('{}/{}'.format(input_training_xml_dir, xml_name))  
                        root = tree.getroot()
                        for elem in root.iter('filename'):  
                                elem.text = image_name.split('.')[0] + '_rotate.jpg'
                        for elem in root.iter('width'):
                                elem.text = str(shape0)
                        for elem in root.iter('height'):
                                elem.text = str(shape1)
                        for elem in root.iter('xmin'):
                                elem.text = str(shape0-ymax)
                        for elem in root.iter('xmax'):
                                elem.text = str(shape0-ymin)
                        for elem in root.iter('ymin'):
                                elem.text = str(xmin)
                        for elem in root.iter('ymax'):
                                elem.text = str(xmax)
                        tree.write('./{}/{}_rotate.xml'.format(output_training_xml_dir, image_name.split('.')[0])) 

                        # crop test
                        cv2.imwrite('test_output/{}'.format('crop_'+image_name.split('.')[0] + '_rotate.jpg'), img1[xmin:xmax, shape0-ymax:shape0-ymin])
                        continue

for image_name in os.listdir(input_testing_dir):
    if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
        img = cv2.imread('{}/{}'.format(input_testing_dir, image_name))
        img1 = imutils.rotate_bound(img, 90)
        cv2.imwrite('./{}/{}.jpg'.format(output_testing_dir, image_name.split('.')[0] + '_rotate'),img1)

        # revise corresponding xml
        for xml_name in os.listdir(input_testing_xml_dir):
                if 'xml' in xml_name and xml_name.split('.')[0] == image_name.split('.')[0]:

                        # read xml
                        xml_doc = minidom.parse('{}/{}'.format(input_testing_xml_dir, xml_name))
                        shape1 = int(xml_doc.getElementsByTagName('width')[0].childNodes[0].data)
                        shape0 = int(xml_doc.getElementsByTagName('height')[0].childNodes[0].data)
                        xmin = int(xml_doc.getElementsByTagName('xmin')[0].childNodes[0].data)
                        xmax = int(xml_doc.getElementsByTagName('xmax')[0].childNodes[0].data)
                        ymin = int(xml_doc.getElementsByTagName('ymin')[0].childNodes[0].data)
                        ymax = int(xml_doc.getElementsByTagName('ymax')[0].childNodes[0].data)

                        # change xml
                        tree = ET.parse('{}/{}'.format(input_testing_xml_dir, xml_name))  
                        root = tree.getroot()
                        for elem in root.iter('filename'):  
                                elem.text = image_name.split('.')[0] + '_rotate.jpg'
                        for elem in root.iter('width'):
                                elem.text = str(shape0)
                        for elem in root.iter('height'):
                                elem.text = str(shape1)
                        for elem in root.iter('xmin'):
                                elem.text = str(shape0-ymax)
                        for elem in root.iter('xmax'):
                                elem.text = str(shape0-ymin)
                        for elem in root.iter('ymin'):
                                elem.text = str(xmin)
                        for elem in root.iter('ymax'):
                                elem.text = str(xmax)
                        tree.write('./{}/{}_rotate.xml'.format(output_testing_xml_dir, image_name.split('.')[0])) 

                        # crop test
                        cv2.imwrite('test_output/{}'.format('crop'+image_name.split('.')[0] + '_rotate.jpg'), img1[xmin:xmax, shape0-ymax:shape0-ymin])
                        continue