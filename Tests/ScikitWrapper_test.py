"""
Tests for functionality in the ScikitWrapper module.
"""

import pytest

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

from TFHelpers.ScikitWrapper import SKTFWrapper
from TFHelpers.SKTFModels import BasicRegressor

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
                               [10, 5])
        model.fit(X_train, y_train, X_val, y_val, 5)

        y_pred = model.predict(X_val)

        assert mean_squared_error(y_val, y_pred) == pytest.approx(38300, 300)
