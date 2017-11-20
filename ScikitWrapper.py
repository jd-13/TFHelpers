import sys
import time

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

import tensorflow as tf

from FilesAndLogging import CheckpointAndRestoreHelper, FileManager, TensorboardLogHelper
from TrainingHelpers import EarlyStoppingHelper, ProgressCalculator

class SKTFWrapper(BaseEstimator, RegressorMixin):
    """
    Doesn't actually do anything, just provides some common functionality used for wrapping TF
    models in an sklearn API.

    You must implement the constructor, fit, and predict methods.
    """

    _session = None

    def fit(self, X, y, X_valid, y_valid, numEpochs=1):
        """
        Build and train the graph here

        ** Derived classes should implement this **
        """
        raise NotImplementedError()

    def predict(self, X):
        """
        Return predictions here
        
        ** Derived classes should implement this **
        """
        raise NotImplementedError()

    def _closeSession(self):
        """Ends the tensorflow session if one is open"""
        if self._session:
            self._session.close()

class RegressorTensors:
    """
    Derived classes of TFRegressor must provide this member, it is the interface between the
    training loop of TFRegressor and the derived class
    """
    def __init__(self,
                 X_in,
                 y_in,
                 logits,
                 loss,
                 trainingOp,
                 init,
                 saver,
                 dropoutKeepProb):
        self.X_in, self.y_in = X_in, y_in
        self.logits, self.loss = logits, loss
        self.trainingOp = trainingOp
        self.init, self.saver = init, saver
        self.dropoutKeepProb = dropoutKeepProb

class TFRegressor(SKTFWrapper):
    """
    Provides functionality that is common to TF regression models, mainly the training loop.

    Derived classes must:
    - Provide a constructor which calls the constructor of this class
    - Implement a _buildGraph method which assigns a RegressorTensors object to the _tensors member
    - Implement a _buildModelName method which returns a url/filename safe string describing the
      model type and its hyperparameters
    """

    def __init__(self,
                 learningRate,
                 batchSize,
                 initializer,
                 dropoutRate,
                 restoreFrom):

        # Scikit-learn's api demands that parameters in the constructor are assigned to members with
        # exactly the same name otherwise its clone method sets everything to None
        # (see BaseEstimator::get_params)
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.initializer = initializer
        self.dropoutRate = dropoutRate
        self.restoreFrom = restoreFrom

        self._session = None
        self._graph = tf.Graph()

        self._fileManager = FileManager(self._buildModelName(), restoreFrom)
        self._allowRestore = restoreFrom is not None

        self._tensors = None

        self._previousTrainables = None

    def _buildGraph(self, numFeatures):
        """
        Build the graph and set the _tensors member to a RegressorTensors object which contains the
        important tensors for the graph

        ** Derived classes should implement this **
        """
        raise NotImplementedError()

    def _buildModelName(self):
        """
        Return a url/filename safe string describing the model type and its hyperparameters

        ** Derived classes should implement this **
        """
        raise NotImplementedError()

    def fit(self, X, y, X_valid, y_valid, numEpochs=1):
        """Fits the model on the training set"""
        self._closeSession()

        with self._graph.as_default():
            self._buildGraph(X.shape[1])

            tensorboardHelper = TensorboardLogHelper(self._fileManager.getModelDir(),
                                                     tf.get_default_graph(),
                                                     ["LossTrain", "LossVal", "BatchTimeAvg"])

        stoppingHelper = EarlyStoppingHelper()
        progressCalc = ProgressCalculator(numEpochs)
        restoreHelper = CheckpointAndRestoreHelper(self._fileManager.getModelDirAndPrefix(),
                                                   self._tensors.saver)

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            if self._allowRestore:
                startEpoch = restoreHelper.restoreIfCheckpoint(sess)
            else:
                startEpoch = 0

            self._tensors.init.run()

            progressCalc.start()
            for epoch in range(startEpoch, numEpochs):
                randomIndicies = np.random.permutation(len(X))
                NUM_BATCHES = len(X) // self.batchSize

                batchTimes = []
                for batchNumber, batchIndicies in enumerate(np.array_split(randomIndicies, NUM_BATCHES)):
                    batchStart = time.time()

                    X_batch, y_batch = X[batchIndicies], y[batchIndicies]

                    feed_dict = {self._tensors.X_in: X_batch,
                                 self._tensors.y_in: y_batch,
                                 self._tensors.dropoutKeepProb: 1 - self.dropoutRate}

                    sess.run(self._tensors.trainingOp, feed_dict=feed_dict)

                    batchTimes.append(time.time() - batchStart)
                    print("Batch:", batchNumber, "/", NUM_BATCHES, "{0:.4f}".format(batchTimes[-1]) + "s", end="\r")

                # Calculate and log the losses for this epoch
                lossTrain = self._tensors.loss.eval(feed_dict=feed_dict)
                lossVal = self._evalLossBatched(X_valid, y_valid)
                tensorboardHelper.writeSummary(sess, [lossTrain, lossVal, np.average(batchTimes)])
                progressCalc.updateInterval(1)
                print("Epoch:", epoch, "Validation loss:", lossVal, "Time Remaining:", progressCalc.getTimeStampRemaining())

                restoreHelper.saveCheckpoint(sess, epoch)

                # Check if all variables are being trained
                if epoch < 2:
                    currentTrainables = sess.run(
                        self._graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

                    if self._previousTrainables is not None:
                        for previousVariable, currentVariable in zip(self._previousTrainables, currentTrainables):
                            if (previousVariable == currentVariable).any():
                                raise RuntimeError("Not all variables have been trained")

                    self._previousTrainables = currentTrainables

                # This must be the last thing done in an epoch
                if stoppingHelper.shouldStop(lossVal):
                    print("Early stopping at epoch: ", epoch)
                    break

            stoppingHelper.restoreBestModelParams()
            tensorboardHelper.close()

        print("Time taken:", progressCalc.timeTaken())

    def predict(self, X):
        """Returns the model's predictions for the provided data"""
        if not self._session:
            raise NotFittedError("This", self.__class__.__name__, "instance is not fitted yet")

        with self._session.as_default():
            return self._tensors.logits.eval(feed_dict={self._tensors.X_in: X})

    def _evalLossBatched(self, X, y):
        """Do validation in batches in case the dataset would need 10's of GB"""
        NUM_BATCHES = len(X) // self.batchSize
        indicies = np.arange(len(X))
        losses = np.zeros(len(X))

        for batchIndicies in np.array_split(indicies, NUM_BATCHES):
            X_batch = X[batchIndicies, :]
            y_batch = y[batchIndicies]

            losses[batchIndicies] = self._tensors.loss.eval(feed_dict={self._tensors.X_in: X_batch,
                                                                       self._tensors.y_in: y_batch})

        return np.average(losses)

    def _mapInitializerName(self):
        """Maps initializer types to a short string suitable for the model's file name"""

        initString = "None"
        if "variance_scaling_initializer" in str(self.initializer):
            initString = "he"
        elif "xavier_initializer" in str(self.initializer):
            initString = "xa"

        return initString
