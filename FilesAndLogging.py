# standard libs
import os

# tensorflow
import tensorflow as tf

class CheckpointAndRestoreHelper:
    """
    Provides functionality for saving the model during training (.ckpt) and writing the number of
    epochs to a file.
    """
    def __init__(self, modelRootPath, saver):
        self.MODEL_CKPT_PATH = modelRootPath + ".ckpt"
        self.MODEL_EPOCH_PATH = self.MODEL_CKPT_PATH + ".epoch"
        self._saver = saver

    def restoreIfCheckpoint(self, sess):
        """Attempts to restore from a checkpoint if one is available. Returns the epoch number to
        start from"""
        if os.path.isfile(self.MODEL_EPOCH_PATH):
            with open(self.MODEL_EPOCH_PATH, "rb") as f:
                startEpoch = int(f.read())

            print("Found checkpoint file, continuing from epoch:", startEpoch)
            self._saver.restore(sess, self.MODEL_CKPT_PATH)
        else:
            print("No checkpoint found, using new model")
            startEpoch = 0

        return startEpoch

    def saveCheckpoint(self, sess, epoch):
        """Saves the model with the .ckpt extension and updates the epoch counter."""
        self._saver.save(sess, self.MODEL_CKPT_PATH)
        with open(self.MODEL_EPOCH_PATH, "wb") as f:
            f.write(b"%d" % (epoch + 1))

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
    def __init__(self, logDir, graph, summaryNames):
        self._fileWriter = tf.summary.FileWriter(logDir, graph)

        self._summaryPlaceholders = [tf.placeholder(shape=(), dtype=tf.float32)
                                     for _ in summaryNames]
        self._summaries = [tf.summary.scalar(name, placeholder)
                           for name, placeholder in zip(summaryNames, self._summaryPlaceholders)]

        self._summaries = tf.summary.merge_all()

        self._iteration = 0

    def writeSummary(self, sess, summaryValues):
        """
        Provide a session and a list of values that correspond to the summary names this object
        was initialised with.
        """
        assert len(summaryValues) == len(self._summaryPlaceholders)

        feed_dict = {placeholder: value
                     for placeholder, value in zip(self._summaryPlaceholders, summaryValues)}

        summaryStrs = sess.run(self._summaries, feed_dict=feed_dict)
        self._fileWriter.add_summary(summaryStrs, self._iteration)

        self._iteration += 1

    def close(self):
        """Close the file writer"""
        self._fileWriter.close()
