import datetime
import time

import numpy as np

import tensorflow as tf

class EarlyStoppingHelper:
    """Contains most of the functionality needed to implement early stopping."""

    MAX_CHECKS_WITHOUT_PROGRESS = 20

    bestLossVal = np.infty
    checksSinceLastProgress = 0
    bestModelParams = None

    def __init__(self, maxChecksWithoutProgress=20):
        self.MAX_CHECKS_WITHOUT_PROGRESS = maxChecksWithoutProgress

    def getModelParams(self):
        """Returns a dictionary of tf.GraphKeys.GLOBAL_VARIABLES"""
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value
                for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def restoreBestModelParams(self):
        """If we have the best model parameters saved, this will restore them."""
        success = False

        if self.bestModelParams:
            gvar_names = list(self.bestModelParams.keys())
            assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                          for gvar_name in gvar_names}

            init_values = {gvar_name: assign_op.inputs[1]
                           for gvar_name, assign_op in assign_ops.items()}
            feed_dict = {init_values[gvar_name]: self.bestModelParams[gvar_name]
                         for gvar_name in gvar_names}
            tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

            success = True

        return success

    def shouldStop(self, lossVal):
        """Returns True if you should stop. Check this at the end of each epoch."""

        if lossVal < self.bestLossVal:
            self.bestLossVal = lossVal
            self.checksSinceLastProgress = 0
            self.bestModelParams = self.getModelParams()
        else:
            self.checksSinceLastProgress += 1

        return self.checksSinceLastProgress > self.MAX_CHECKS_WITHOUT_PROGRESS

class ProgressCalculator:
    """
    Provided a number of iterations, if updateInterval is called on completion of each iteration
    then the time to complete the remaining iterations is calculated.
    """
    _itersPerSecond = 0
    _progress = 0
    _target = 0
    _startEpoch = 0

    def __init__(self, targetCount):
        self._target = targetCount

    def start(self):
        """Records the current time as the time when the first iteration started."""
        self._startEpoch = time.time()

    def reset(self):
        """Reset the timer and all internal counters"""
        self._itersPerSecond = 0
        self._progress = 0
        self._target = 0
        self._startEpoch = 0

    def updateInterval(self, newlyCompletedIterations):
        """
        Increment the completed iterations by the provided amount and recalculate the time per
        iteration.
        """
        self._progress += newlyCompletedIterations

        timeSinceStart = time.time() - self._startEpoch
        self._itersPerSecond = self._progress / timeSinceStart

    def getSecondsRemaining(self):
        """Predict the number of seconds to complete the remaining iterations"""
        try:
            return (self._target - self._progress) / self._itersPerSecond
        except ZeroDivisionError:
            return 0

    def getTimeStampRemaining(self):
        """
        Predict the time to complete the remaining iterations and return a formatted timestamp.
        """
        timestamp = datetime.timedelta(seconds=self.getSecondsRemaining())
        return str(timestamp)

    def timeTaken(self):
        """Returns a formatted timestamp of the total time since start was called."""
        timeSinceStart = time.time() - self._startEpoch
        timestamp = datetime.timedelta(seconds=timeSinceStart)
        return str(timestamp)
