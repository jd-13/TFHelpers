"""
Tests for functionality in the ScikitWrapper module.
"""

import pytest

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

from TFHelpers.ScikitWrapper import SKTFWrapper, RegressorTensors, TFRegressor

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
                 outputLength):

        TFRegressor.__init__(self,
                             learningRate,
                             batchSize,
                             initializer,
                             dropoutRate,
                             restoreFrom,
                             outputLength)

    def _buildGraph(self, numFeatures):
        """Builds the graph using the default graph"""

        with tf.name_scope("inputs"):
            X_in = tf.placeholder(shape=(None, numFeatures), dtype=tf.float32, name="X_in")
            y_in = tf.placeholder(shape=(None), dtype=tf.float32, name="y_in")

        with tf.name_scope("dnn"):
            dropoutKeepProb = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
            layer1 = tf.layers.dense(X_in, 10, kernel_initializer=self.initializer)
            logits = tf.layers.dense(layer1, units=self.outputLength, kernel_initializer=self.initializer)

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
        return self.__class__.__name__

class Test_SKTFWrapper:
    """
    Tests for the FileManager class.
    """
    def test_NotImplementedMethods(self):
        """
        Tests if the correct model paths are generated if no model to restore from is provided.
        """

        wrapper = SKTFWrapper()

        with pytest.raises(NotImplementedError):
            wrapper.fit(None, None, None, None)

        with pytest.raises(NotImplementedError):
            wrapper.predict(None)

    def test_BasicRegressor(self):
        """
        Train a TFRegressor model and test the accuracy.
        """
        X, y = make_regression(1000, 20, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
        tf.set_random_seed(42)

        model = BasicRegressor(0.01,
                               100,
                               tf.contrib.layers.variance_scaling_initializer(),
                               0.1,
                               None,
                               1)
        model.fit(X_train, y_train, X_val, y_val, 5)

        y_pred = model.predict(X_val)

        assert mean_squared_error(y_val, y_pred) == pytest.approx(38300, 300)
