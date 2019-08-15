from xml.dom import minidom
import cv2
import os

XML_dir = 'training_labels_xml' # 'training_labels_xml' or 'testing_labels_xml'
IMAGE_dir = 'training_images' # 'training_images' or 'testing_images'
EXPORT_dir = 'crop_training' # 'crop_training' or 'cropc_testing'

for xml_name in os.listdir(XML_dir):
    if 'xml' in xml_name:
        xml_doc = minidom.parse('{}/{}'.format(XML_dir, xml_name))
        xmin = int(xml_doc.getElementsByTagName('xmin')[0].childNodes[0].data)
        xmax = int(xml_doc.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymin = int(xml_doc.getElementsByTagName('ymin')[0].childNodes[0].data)
        ymax = int(xml_doc.getElementsByTagName('ymax')[0].childNodes[0].data)
        for image_name in os.listdir(IMAGE_dir):
            if "jpg" in image_name or "jpeg" in image_name or "png" in image_name:
                if image_name.split('.')[0] == xml_name.split('.')[0]:
                    image = cv2.imread('{}/{}'.format(IMAGE_dir, image_name))
                    image = image[ymin:ymax, xmin:xmax]
                    cv2.imwrite('./{}/{}.jpg'.format(EXPORT_dir, image_name.split('.')[0]), image)
                    break