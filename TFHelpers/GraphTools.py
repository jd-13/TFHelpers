"""
Tools for manipulating model and graph files outside of training/inference.
"""

import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

def metaToProtoBufGraph(modelDir: str, modelBaseName: str, outFileName: str) -> None:
    """
    Converts model files to a protobuf graph def.
    """
    saver = tf.train.import_meta_graph(os.path.join(modelDir, modelBaseName + ".ckpt.meta"))

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(str(modelDir)))
        tf.train.write_graph(sess.graph_def, str(modelDir), outFileName, as_text=True)

def freezeProtoBufGraph(modelDir: str,
                        modelBaseName: str,
                        outputNodeName: str,
                        inFileName: str,
                        outFileName: str) -> None:
    """
    Freezes variable values into a protobuf graph def.
    """
    freeze_graph.freeze_graph(input_graph=os.path.join(modelDir, inFileName),
                              input_saver=None,
                              input_binary=False,
                              input_checkpoint=os.path.join(modelDir, modelBaseName + ".ckpt"),
                              output_node_names=outputNodeName,
                              restore_op_name="",
                              filename_tensor_name="",
                              output_graph=os.path.join(modelDir, outFileName),
                              clear_devices=False,
                              initializer_nodes="",
                              variable_names_whitelist="",
                              variable_names_blacklist="",
                              input_meta_graph=None,
                              input_saved_model_dir=None)

def removeTrainingNodesFromProtoBufGraph(modelDir: str,
                                         inputNodeName: str,
                                         outputNodeName: str,
                                         inFileName: str,
                                         outFileName: str,
                                         inputShape: str,
                                         tensorflowPath: str):
    """
    Removes training nodes from a protobuf graph def.

    modelDir must be an absolute path.
    """

    TRANSFORM_PATH = "tensorflow/tools/graph_transforms:transform_graph --"
    IN_GRAPH = "--in_graph={}".format(os.path.join(modelDir, inFileName))
    OUT_GRAPH = "--out_graph={}".format(os.path.join(modelDir, outFileName))
    INPUTS = "--inputs='{}'".format(inputNodeName)
    OUTPUTS = "--outputs='{}'".format(outputNodeName)
    TRANFORMS = "--transforms='strip_unused_nodes(type=float, shape=\"{}\")'".format(inputShape)

    os.system("cd {} && bazel run {} {} {} {} {} {}".format(tensorflowPath,
                                                            TRANSFORM_PATH,
                                                            IN_GRAPH,
                                                            OUT_GRAPH,
                                                            INPUTS,
                                                            OUTPUTS,
                                                            TRANFORMS))