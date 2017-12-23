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
                 learningRate,
                 batchSize,
                 initializer,
                 dropoutRate,
                 restoreFrom,
                 hiddenNeuronsList):

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
