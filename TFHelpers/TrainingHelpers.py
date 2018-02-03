"""
Utilities that are useful during the training phase, such calculating the time remaining and early
stopping.
"""
import datetime
import time
from typing import Any, Dict
import warnings

import numpy as np

import tensorflow as tf

class EarlyStoppingHelper:
    """Contains most of the functionality needed to implement early stopping."""

    def __init__(self, maxChecksWithoutProgress: int=20):
        self.MAX_CHECKS_WITHOUT_PROGRESS = maxChecksWithoutProgress

        self.bestLossVal = np.infty
        self.checksSinceLastProgress = 0
        self.bestModelParams = None

    def _getModelParams(self) -> Dict[str, Any]:
        """Returns a dictionary of tf.GraphKeys.GLOBAL_VARIABLES"""
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value
                for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def restoreBestModelParams(self) -> bool:
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

    def shouldStop(self, lossVal: float) -> bool:
        """Returns True if you should stop. Check this at the end of each epoch."""

        if lossVal < self.bestLossVal:
            self.bestLossVal = lossVal
            self.checksSinceLastProgress = 0
            self.bestModelParams = self._getModelParams()
        else:
            self.checksSinceLastProgress += 1

        return self.checksSinceLastProgress > self.MAX_CHECKS_WITHOUT_PROGRESS

class ProgressCalculator:
    """
    Provided a number of iterations, if updateInterval is called on completion of each iteration
    then the time to complete the remaining iterations is calculated.
    """

    def __init__(self, targetIterations: float):
        if targetIterations < 1:
            raise ValueError("targetIterations must larger than zero")

        self._targetIterations = targetIterations
        self._itersPerSecond = 0
        self._completedIterations = 0
        self._startEpochTime = 0

    def start(self) -> None:
        """Records the current time as the time when the first iteration started."""
        self._startEpochTime = time.time()

    def reset(self) -> None:
        """Reset the timer and all internal counters"""
        self._itersPerSecond = 0
        self._completedIterations = 0
        self._startEpochTime = 0

    def updateInterval(self, newlyCompletedIterations: float) -> None:
        """
        Increment the completed iterations by the provided amount and recalculate the time per
        iteration.
        """
        if self._startEpochTime is 0:
            raise RuntimeError("start must be called before attempting to increment this counter")

        self._completedIterations += newlyCompletedIterations

        timeSinceStart = time.time() - self._startEpochTime
        self._itersPerSecond = self._completedIterations / timeSinceStart

    def getSecondsRemaining(self) -> float:
        """Predict the number of seconds to complete the remaining iterations"""
        if self._completedIterations is 0:
            raise RuntimeError(
                "updateInterval must be called before attempting to evaluate the time remaining")

        try:
            return (self._targetIterations - self._completedIterations) / self._itersPerSecond
        except ZeroDivisionError:
            return 0.0

    def getTimeStampRemaining(self) -> str:
        """
        Predict the time to complete the remaining iterations and return a formatted timestamp.
        """
        timestamp = datetime.timedelta(seconds=self.getSecondsRemaining())
        return str(timestamp)

    def timeTaken(self) -> str:
        """Returns a formatted timestamp of the total time since start was called."""
        if self._startEpochTime is 0:
            raise RuntimeError("start must be called before attempting to evaluate the time taken")

        timeSinceStart = time.time() - self._startEpochTime
        timestamp = datetime.timedelta(seconds=timeSinceStart)
        return str(timestamp)

class TrainingValidator:
    """
    Performs simple checks on the model during training to ensure that the model is training
    correctly.
    """
    def __init__(self, graph, session):
        self._graph = graph
        self._session = session
        self._previousTrainables = None
    
    def validate(self, loss: float) -> None:
        """
        Call this once in each epoch, though you may wish to only do this for the first few epochs.
        Warnings are raised for each check that fails.
        """
        self._checkVariablesTrained()
        self._checkLoss(loss)
        self._checkStdDeviation()

    def _checkVariablesTrained(self) -> None:
        """
        Compares the values of all TRAINABLE_VARIABLES in this epoch to the previous epoch. Raises
        a warning if they are the same.
        """
        variableNames = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        currentTrainables = self._session.run(variableNames)

        if self._previousTrainables is not None:
            for previousVariable, currentVariable, name in zip(self._previousTrainables, currentTrainables, variableNames):
                if (previousVariable == currentVariable).any():
                    warnings.warn("The following variable might not have been trained: {0}".format(name),
                                  RuntimeWarning)

        self._previousTrainables = currentTrainables

    def _checkLoss(self, loss: float) -> None:
        """
        Checks that the loss is a valid value.
        """
        if loss == 0:
            warnings.warn("Loss is currently zero, this is likely to be an error",
                          RuntimeWarning)
    
    def _checkStdDeviation(self) -> None:
        """
        Most trainable variables should have a standard deviation below 2. Anything above this may
        indicate an exploding gradient.
        """
        variableNames = self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainables = self._session.run(variableNames)

        for name, variable in zip(variableNames, trainables):
            stdDev = variable.std() 
            if stdDev > 2:
                warnings.warn("Variable {0} has a standard deviation of {1}. This may indicate a vanishing/exploding gradient".format(name, stdDev),
                              RuntimeWarning)
