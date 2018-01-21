"""
Utilities for writing and managing files for models and their logs.
"""

from datetime import datetime
import pathlib
from typing import List

import tensorflow as tf

class CheckpointAndRestoreHelper:
    """
    Provides functionality for saving the model during training (*.ckpt.*) and writing the number of
    epochs to a file.
    """
    def __init__(self, modelRootPath: str, shouldRestore: bool, graph):
        self.MODEL_ROOT_DIR = str(pathlib.Path(modelRootPath).parents[0])
        self.MODEL_CKPT_PATH = modelRootPath + ".ckpt"
        self.MODEL_EPOCH_PATH = self.MODEL_CKPT_PATH + ".epoch"
        MODEL_META_PATH = self.MODEL_CKPT_PATH +".meta"

        with graph.as_default():
            if shouldRestore:
                self._saver = tf.train.import_meta_graph(MODEL_META_PATH)
            else:
                self._saver = tf.train.Saver()

    def restoreFromCheckpoint(self, sess) -> int:
        """Attempts to restore from a checkpoint if one is available. Returns the epoch number to
        start from"""
        with open(self.MODEL_EPOCH_PATH, "rb") as f:
            startEpoch = int(f.read())

        print("Found checkpoint file, continuing from epoch:", startEpoch)
        self._saver.restore(sess, tf.train.latest_checkpoint(self.MODEL_ROOT_DIR))

        return startEpoch

    def saveCheckpoint(self, sess, epoch: int) -> None:
        """Saves the model with the .ckpt extension and updates the epoch counter."""
        self._saver.save(sess, self.MODEL_CKPT_PATH)
        with open(self.MODEL_EPOCH_PATH, "wb") as f:
            f.write(b"%d" % (epoch + 1))

class FileManager:
    """
    Simple class to manage where model files will be written.
    Creates/expects a folder structure that looks like:
    ./models
    ./models/<modelName>
    ./models/<modelName>/<datetime>
    ./models/<modelName>/<datetime>/model*
    """
    def __init__(self, modelName: str, restoreFrom: str=None):
        """
        Provide restoreFrom to reuse an existing model, this folder should contain the model files
        in the format:
            model.final.data-00000-of-00001, model.final.index, model.final.meta
        """

        if restoreFrom is None:
            self._modelDir = pathlib.Path.cwd() / "models" / modelName / datetime.utcnow().strftime("%Y%m%d-%H%M")
            self._modelDir.mkdir(parents=True, exist_ok=True)
        else:
            self._modelDir = pathlib.Path.cwd() / "models" / modelName / restoreFrom

    def getModelDir(self) -> str:
        """
        Returns the directory of the model
        eg: models/Seq2SeqRegressor-X-200-H-50_30_10-I-he-D-None/20171116-2329
        """
        return str(self._modelDir)

    def getModelDirAndPrefix(self) -> str:
        """
        Returns the model directory with the file prefix
        eg: models/Seq2SeqRegressor-X-200-H-50_30_10-I-he-D-None/20171116-2329/model
        """
        return str(self._modelDir / "model")

class TensorboardLogHelper:
    """
    Writes summaries to log files for tensorboard.

    Provide a list of strings of the names of summaries that you will provide the values for when
    calling addSummary. This is done to allow the same node on the graph to produce several
    different summaries, eg. the loss when feed_dict has training data and when feed_dict has
    validation data.

    Does a tf.summary.merge_all so any other summaries added to the graph will also get written to
    the log when writeSummary is called.
    """
    def __init__(self, logDir, graph, summaryNames: List[str]):
        self._fileWriter = tf.summary.FileWriter(logDir, graph)

        self._summaryPlaceholders = [tf.placeholder(shape=(), dtype=tf.float32)
                                     for _ in summaryNames]
        self._summaries = [tf.summary.scalar(name, placeholder)
                           for name, placeholder in zip(summaryNames, self._summaryPlaceholders)]

        self._summaries = tf.summary.merge_all()

        self._iteration = 0

    def writeSummary(self, sess, summaryValues: List[float]) -> None:
        """
        Provide a session and a list of values that correspond to the summary names this object
        was initialised with.
        """
        if len(summaryValues) is not len(self._summaryPlaceholders):
            raise ValueError("Number of summary values provided ({0}) does not match the number of,\
                              summary names ({1}) this object was initialised with".format(
                                  len(summaryValues), len(self._summaryPlaceholders)))

        feed_dict = {placeholder: value
                     for placeholder, value in zip(self._summaryPlaceholders, summaryValues)}

        summaryStrs = sess.run(self._summaries, feed_dict=feed_dict)
        self._fileWriter.add_summary(summaryStrs, self._iteration)

        self._iteration += 1

    def close(self) -> None:
        """Close the file writer"""
        self._fileWriter.close()
