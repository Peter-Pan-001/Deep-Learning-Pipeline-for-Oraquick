import tensorflow as tf
import numpy as np
from create_tf_record import *
from tensorflow.python.framework import graph_util
 
resize_height = 299  # 指定图片高度
resize_width = 299  # 指定图片宽度
depths = 3

def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
 
            # 定义输入的张量名称,对应网络结构的输入张量
            # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
            input_image_tensor = sess.graph.get_tensor_by_name("input:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("keep_prob:0")
            input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
 
            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("InceptionV3/Logits/SpatialSqueeze:0")
 
            # 读取测试图片
            im=read_image(image_path,resize_height,resize_width,normalization=True)
            im=im[np.newaxis,:]
            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            # out=sess.run("InceptionV3/Logits/SpatialSqueeze:0", feed_dict={'input:0': im,'keep_prob:0':1.0,'is_training:0':False})
            out=sess.run(output_tensor_name, feed_dict={input_image_tensor: im,
                                                        input_keep_prob_tensor:1.0,
                                                        input_is_training_tensor:False})
            print("out:{}".format(out))
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            print("pre class_id:{}".format(sess.run(class_id)))

image_path = 'test_image/animal.jpg'
out_pb_path = ''
freeze_graph_test(pb_path=out_pb_path, image_path=image_path)