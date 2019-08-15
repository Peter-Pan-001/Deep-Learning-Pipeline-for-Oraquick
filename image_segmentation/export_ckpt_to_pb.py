import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(input_checkpoint,output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: 
    :return:
    '''
    output_node_names = "conv2d_15/Sigmoid"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

input_checkpoint='segmentation_model_001/model.ckpt-1880'
out_pb_path="saved_model/saved_model.pb"
freeze_graph(input_checkpoint,out_pb_path)