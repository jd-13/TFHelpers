"""
Example implementations of tensorflow models using the Scikit Wrapper.
"""

import tensorflow as tf

from TFHelpers.ScikitWrapper import RegressorTensors, TFRegressor

class BasicRegressor(TFRegressor):
    """
    Simple regression model for testing TFRegressor.
    """
    def __init__(self,
                 learningRate=0.01,
                 batchSize=1000,
                 initializer=tf.contrib.layers.variance_scaling_initializer(),
                 dropoutRate=0.01,
                 restoreFrom=None,
                 hiddenNeuronsList=[10]):

        self.hiddenNeuronsList = hiddenNeuronsList

        TFRegressor.__init__(self,
                             learningRate,
                             batchSize,
                             initializer,
                             dropoutRate,
                             restoreFrom,
                             1)

    def _buildGraph(self, numFeatures):
        """Builds the graph using the default graph"""

        with tf.name_scope("inputs"):
            X_in = tf.placeholder(shape=(None, numFeatures), dtype=tf.float32, name="X_in")
            y_in = tf.placeholder(shape=(None), dtype=tf.float32, name="y_in")

        with tf.name_scope("dnn"):
            # We don't implement dropout in this model, but we need to provide a placeholder for it
            # anyway
            dropoutKeepProb = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")

            layerOutput = X_in
            for numNeurons in self.hiddenNeuronsList:
                layerOutput = tf.layers.dense(layerOutput,
                                              numNeurons,
                                              kernel_initializer=self.initializer)

            # Create histogram summaries
            for layer in range(len(self.hiddenNeuronsList)):
                path = "dense" if layer == 0 else "dense_{0}".format(layer)

                tf.summary.histogram("dense_{0}/kernel".format(layer),
                                     tf.get_default_graph().get_tensor_by_name(path + "/kernel:0"))
                tf.summary.histogram("dense_{0}/bias".format(layer),
                                     tf.get_default_graph().get_tensor_by_name(path + "/bias:0"))

            logits = tf.layers.dense(layerOutput,
                                     units=1,
                                     kernel_initializer=self.initializer,
                                     name="logits")

        with tf.name_scope("loss"):
            mse = tf.reduce_mean(tf.square(logits - y_in), name="mse")

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
            trainingOp = optimizer.minimize(mse, name="trainingOp")

        return RegressorTensors(X_in,
                                y_in,
                                logits,
                                mse,
                                trainingOp,
                                dropoutKeepProb)

    def _restoreGraph(self, graph):
        return RegressorTensors(graph.get_tensor_by_name("inputs/X_in:0"),
                                graph.get_tensor_by_name("inputs/y_in:0"),
                                graph.get_tensor_by_name("dnn/logits:0"),
                                graph.get_tensor_by_name("loss/mse:0"),
                                graph.get_operation_by_name("train/trainingOp"),
                                graph.get_tensor_by_name("dnn/keep_prob:0"))

    def _buildHyperParamsDict(self):
        return {"H": "_".join(str(value) for value in self.hiddenNeuronsList),
                "I": self._mapInitializerName(self.initializer),
                "D": str(self.dropoutRate)}
