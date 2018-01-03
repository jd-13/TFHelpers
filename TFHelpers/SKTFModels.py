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

            logits = tf.layers.dense(layerOutput, units=1, kernel_initializer=self.initializer)

        with tf.name_scope("loss"):
            mse = tf.reduce_mean(tf.square(logits - y_in), name="mse")

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
            trainingOp = optimizer.minimize(mse)

        with tf.name_scope("init_and_save"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        self._tensors = RegressorTensors(X_in,
                                         y_in,
                                         logits,
                                         mse,
                                         trainingOp,
                                         init,
                                         saver,
                                         dropoutKeepProb)

    def _buildModelName(self):
        return self.__class__.__name__ + "-H-" + "_".join(str(value)
                                                          for value in self.hiddenNeuronsList) \
                                       + "-I-" + self._mapInitializerName() \
                                       + "-D-" + str(self.dropoutRate)
