"""
Tests for functionality in the TrainingHelpers module.
"""

import time
import pytest

from sklearn.datasets import make_regression

import tensorflow as tf

from TFHelpers.TrainingHelpers import ProgressCalculator, EarlyStoppingHelper, TrainingValidator

class Test_ProgressCalculator_InvalidBehaviour:
    """
    Tests that the ProgressCalculator behaves appropriately when operated incorrectly.
    """
    def test_InvalidIterations(self):
        """
        ProgressCalculator should raise an exception if initialised with an invalid number of
        iterations.
        """
        with pytest.raises(ValueError):
            ProgressCalculator(-10)

        with pytest.raises(ValueError):
            ProgressCalculator(0)

    def test_TimeTakenBeforeStart(self):
        """
        ProgressCalculator should raise an exception if timeTaken is called before the timer has
        been started.
        """
        progress = ProgressCalculator(10)
        with pytest.raises(RuntimeError):
            progress.timeTaken()

    def test_GetTimeRemainingBeforeStart(self):
        """
        ProgressCalculator should raise an exception if the time remaining is retrieved before the
        timer has been started.
        """
        progress = ProgressCalculator(10)

        with pytest.raises(RuntimeError):
            progress.getTimeStampRemaining()

        with pytest.raises(RuntimeError):
            progress.getSecondsRemaining()

    def test_GetTimeRemainingBeforeIncrement(self):
        """
        ProgressCalculator should raise an exception if the time remaining is retrieved before the
        timer has been incremented.
        """
        progress = ProgressCalculator(10)
        progress.start()

        with pytest.raises(RuntimeError):
            progress.getTimeStampRemaining()

        with pytest.raises(RuntimeError):
            progress.getSecondsRemaining()

    def test_GetAttributesAfterReset(self):
        """
        ProgressCalculator should raise an exception if any attributes are retrieved immediately
        after reset.
        """
        progress = ProgressCalculator(10)
        progress.start()
        progress.reset()

        with pytest.raises(RuntimeError):
            progress.timeTaken()

        with pytest.raises(RuntimeError):
            progress.getTimeStampRemaining()

        with pytest.raises(RuntimeError):
            progress.getSecondsRemaining()

    def test_UpdateBeforeStart(self):
        """
        ProgressCalculator should raise an exception if updateInterval is called before start.
        """
        progress = ProgressCalculator(10)

        with pytest.raises(RuntimeError):
            progress.updateInterval(1)

class Test_ProgressCalculator_Operation:
    """
    Tests that the ProgressCalculator behaves appropriately when under normal operation.
    """

    def test_CorrectTimeTaken(self):
        """
        ProgressCalculator should provide the time taken since start was called.
        """
        progress = ProgressCalculator(10)
        progress.start()

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:01.0"

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:02.0"

        # Check this still works after a reset
        progress.reset()
        progress.start()

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:01.0"

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:02.0"

        # Check that the interval doesn't have any affect
        progress.updateInterval(1)

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:03.0"

    def test_CorrectTimeRemaining_ConstantSpeed(self):
        """
        ProgressCalculator should provide an estimate of the time remaining for a constant time
        interval.
        """
        progress = ProgressCalculator(3)
        progress.start()

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:02.0"
        assert progress.getSecondsRemaining() == pytest.approx(2, 0.1)

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:01.0"
        assert progress.getSecondsRemaining() == pytest.approx(1, 0.1)

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining() == "0:00:00"
        assert progress.getSecondsRemaining() == pytest.approx(0, 0.1)

    def test_CorrectTimeRemaining_VaryingSpeed(self):
        """
        ProgressCalculator should provide an estimate of the time remaining for a varying time
        interval.
        """
        progress = ProgressCalculator(3)
        progress.start()

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:02.0"
        assert progress.getSecondsRemaining() == pytest.approx(2, 0.1)

        time.sleep(2.02)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:01.5"
        assert progress.getSecondsRemaining() == pytest.approx(1.5, 0.1)

        time.sleep(3.03)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining() == "0:00:00"
        assert progress.getSecondsRemaining() == pytest.approx(0, 0.1)

class Test_EarlyStoppingHelper:
    """
    Tests that the EarlyStoppingHelper is able to signal when to stop and restore the correct model
    parameters.
    """

    def test_StopAndRestore_Plateau(self):
        """
        Simulates a decreasing loss for several iterations, then a loss which does not change. The
        helper should signal the need to stop after the correct number of iterations.
        """
        stoppingHelper = EarlyStoppingHelper(3)

        # Lets add some basic operators to the graph
        tf.reset_default_graph()
        A = tf.Variable(10, dtype=tf.float32)
        B = tf.Variable(15, dtype=tf.float32)
        init = tf.global_variables_initializer()

        # And some ops so that we can mess with the variables
        A_mod = A.assign(5)
        B_mod = B.assign(5)

        with tf.Session() as sess:
            init.run()

            assert not stoppingHelper.shouldStop(10)
            assert not stoppingHelper.shouldStop(9)
            assert not stoppingHelper.shouldStop(8)
            assert not stoppingHelper.shouldStop(7)

            # Now modify A while the loss is lowest
            A_mod.eval()
            assert not stoppingHelper.shouldStop(6)
            assert not stoppingHelper.shouldStop(6)

            # And modify B while the loss is the same
            B_mod.eval()
            assert not stoppingHelper.shouldStop(6)
            assert not stoppingHelper.shouldStop(6)
            assert stoppingHelper.shouldStop(6)

            # When we restore the parameters A should retain the modified value, and B should be
            # reset
            stoppingHelper.restoreBestModelParams()
            assert A.eval() == 5
            assert B.eval() == 15

    def test_StopAndRestore_Increasing(self):
        """
        Simulates a decreasing loss for several iterations, then a loss which increases. The
        helper should signal the need to stop after the correct number of iterations.
        """
        stoppingHelper = EarlyStoppingHelper(3)

        # Lets add some basic operators to the graph
        tf.reset_default_graph()
        A = tf.Variable(10, dtype=tf.float32)
        B = tf.Variable(15, dtype=tf.float32)
        init = tf.global_variables_initializer()

        # And some ops so that we can mess with the variables
        A_mod = A.assign(5)
        B_mod = B.assign(5)

        with tf.Session() as sess:
            init.run()

            assert not stoppingHelper.shouldStop(10)
            assert not stoppingHelper.shouldStop(9)
            assert not stoppingHelper.shouldStop(8)
            assert not stoppingHelper.shouldStop(7)

            # Now modify A while the loss is lowest
            A_mod.eval()
            assert not stoppingHelper.shouldStop(6)
            assert not stoppingHelper.shouldStop(7)
            assert not stoppingHelper.shouldStop(8)

            # And modify B while the loss is a little higher
            B_mod.eval()
            assert not stoppingHelper.shouldStop(9)

            assert stoppingHelper.shouldStop(10)

            # When we restore the parameters A should retain the modified value, and B should be
            # reset
            stoppingHelper.restoreBestModelParams()
            assert A.eval() == 5
            assert B.eval() == 15

class Test_TrainingValidator:
    """
    Tests that the TrainingValidator is able to raise the correct warnings.
    """

    def test_VariablesTrained(self):
        # Get some data
        NUM_FEATURES = 10
        X, y = make_regression(1000, NUM_FEATURES, random_state=42)

        # Setup a basic graph
        tf.reset_default_graph()
        X_in = tf.placeholder(shape=(None, NUM_FEATURES), dtype=tf.float32)
        y_in = tf.placeholder(shape=(None), dtype=tf.float32)

        layer1 = tf.layers.dense(X_in, units=NUM_FEATURES)

        # Don't connect the next layer, this is what will cause the warning
        layer2 = tf.layers.dense(X_in, units=NUM_FEATURES)
        layer3 = tf.layers.dense(layer1, units=1)

        mse = tf.reduce_mean(tf.square(layer3 - y_in), name="mse")
        trainingOp = tf.train.AdamOptimizer(learning_rate=0.01).minimize(mse)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            init.run()
            trainingValidator = TrainingValidator(tf.get_default_graph(), sess)

            feedDict = {X_in: X, y_in: y}

            # Manually run the first epoch
            sess.run(trainingOp, feed_dict=feedDict)
            trainingValidator.validate(sess.run(mse, feed_dict=feedDict))

            # Run the second epoch. This should raise a warning
            sess.run(trainingOp, feed_dict=feedDict)
            with pytest.warns(RuntimeWarning):
                trainingValidator.validate(sess.run(mse, feed_dict=feedDict))

    def test_ZeroLoss(self):
        """
        Build a basic graph and then provide a loss of 0 to the validator, which should raise a
        warning.
        """
        # Setup a basic graph
        tf.reset_default_graph()
        NUM_FEATURES = 10
        X_in = tf.placeholder(shape=(None, NUM_FEATURES), dtype=tf.float32)
        y_in = tf.placeholder(shape=(None), dtype=tf.float32)

        layer1 = tf.layers.dense(X_in, units=NUM_FEATURES)

        mse = tf.reduce_mean(tf.square(layer1 - y_in), name="mse")
        trainingOp = tf.train.AdamOptimizer(learning_rate=0.01).minimize(mse)

        init = tf.global_variables_initializer()

        # Create a session and set the loss to 0 in the validator
        with tf.Session() as sess:
            init.run()
            trainingValidator = TrainingValidator(tf.get_default_graph(), sess)

            with pytest.warns(RuntimeWarning):
                trainingValidator.validate(0)

    def test_StdDeviation(self):
        # TODO: not yet sure how to test this effectively
        pass
