"""
1) https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py
2) https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pet_tf_record.py

Script adapted from 1) with minor modifications, and acts as a custom version
of 2). The script takes labeled images defined in CSV format, as well as the
images themselves, and converts them into TensorFlow interpretable "records",
so that they can be trained and evaluated on.

Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=training_labels.csv --output_path=training.record
                              --image_dir=training_images
  # Create test data:
  python generate_tfrecord.py --csv_input=data/training_labels.csv --output_path=testing.record
                              --image_dir=testing_images
"""

# Future imports
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Standard imports
import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from collections import namedtuple

# Tensorflow imports
from object_detection.utils import dataset_util

# Define flags that will be inputted by the user
flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to CSV data')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to image directory')
FLAGS = flags.FLAGS

# TODO: Replace 'TestKit' with the desired label name
def class_to_int(label):
    """
    Function to convert the text label into integer

    @param label (str): String representing detected object, same as in .pbtx
    @return 1 or None: Returns 1 if label is the desired label, or returns 0
    """
    if label == 'TestKit':
        return 1
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    try:
        image = Image.open(io.BytesIO(encoded_jpg))
    except:
        # print(group.filename) # debugging
        return None

    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

if __name__ == '__main__':
    tf.app.run()
