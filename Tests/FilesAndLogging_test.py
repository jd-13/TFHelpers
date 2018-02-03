"""
Tests for functionality in the FilesAndLogging module.
"""

from datetime import datetime
import os
import pathlib
import pytest

import tensorflow as tf

from TFHelpers.FilesAndLogging import CheckpointAndRestoreHelper, FileManager

class Test_CheckpointAndRestoreHelper:
    """
    Tests for the CheckpointAndRestoreHelper class.
    """
    def test_FileCreation(self, request):
        """
        Tests if the correct files are generated when a model is saved.
        """

        # Lets add some basic operators to the graph
        tf.reset_default_graph()
        A = tf.Variable(10, dtype=tf.float32)
        B = tf.Variable(15, dtype=tf.float32)
        init = tf.global_variables_initializer()

        # Setup our helpers
        MODEL_DIR = request.node.name
        START_DATETIME = datetime.utcnow().strftime("%Y%m%d-%H%M")
        fileManager = FileManager(MODEL_DIR, None)
        restoreHelper = CheckpointAndRestoreHelper(fileManager.getModelDirAndPrefix(),
                                                   False,
                                                   tf.get_default_graph())

        # Now we'll save the model (we don't actually need to train anything)
        with tf.Session() as sess:
            init.run()
            restoreHelper.saveCheckpoint(sess, 0)

        # Check all files are created
        for filename in ["checkpoint", "model.ckpt.data-00000-of-00001", "model.ckpt.epoch", "model.ckpt.index", "model.ckpt.meta"]:
            assert os.path.isfile(str(pathlib.Path.cwd() / "models" / MODEL_DIR / START_DATETIME / filename))

    def test_RestoreDuringRun(self, request):
        """
        Tests if the model can be restored during a training run where we continue using the same
        graph rather than loading the meta graph.
        """

        # Lets add some basic operators to the graph
        tf.reset_default_graph()
        A = tf.Variable(10, dtype=tf.float32)
        B = tf.Variable(15, dtype=tf.float32)
        init = tf.global_variables_initializer()

        # And some ops so that we can mess with the variables
        A_mod = A.assign(5)
        B_mod = B.assign(5)

        # Setup our helpers
        fileManager = FileManager(request.node.name, None)
        restoreHelper = CheckpointAndRestoreHelper(fileManager.getModelDirAndPrefix(),
                                                   False,
                                                   tf.get_default_graph())

        # Now we'll save the model (we don't actually need to train anything)
        with tf.Session() as sess:
            init.run()
            restoreHelper.saveCheckpoint(sess, 0)

            # Now modify the variables
            A_mod.eval()
            B_mod.eval()
            assert A.eval() == 5
            assert B.eval() == 5

            # Now restore the model and check that the variables are restored
            restoreHelper.restoreFromCheckpoint(sess)
            assert A.eval() == 10
            assert B.eval() == 15

    def test_RestoreFreshRun(self, request):
        """
        Tests if the model can be restored during a training run where we reload the meta graph.
        This would be the most common use case, where the process was been interrupted and we have
        to load the graph from files.
        """

        # Lets add some basic operators to the graph
        tf.reset_default_graph()
        A = tf.Variable(10, dtype=tf.float32, name="A")
        B = tf.Variable(15, dtype=tf.float32, name="B")
        init = tf.global_variables_initializer()

        # And some ops so that we can mess with the variables
        A_mod = A.assign(5)
        B_mod = B.assign(5)

        # Setup our helpers
        fileManager = FileManager(request.node.name, None)
        restoreHelper = CheckpointAndRestoreHelper(fileManager.getModelDirAndPrefix(),
                                                   False,
                                                   tf.get_default_graph())

        # Now we'll save the model (we don't actually need to train anything)
        with tf.Session() as sess:
            init.run()
            restoreHelper.saveCheckpoint(sess, 0)

            # Now modify the variables
            A_mod.eval()
            B_mod.eval()
            assert A.eval() == 5
            assert B.eval() == 5

        # Now reset the graph and check that the variables are restored after loading the meta
        # graph. To do this we create CheckpointAndRestoreHelper with shouldRestore=True.
        tf.reset_default_graph()
        restoreHelper = CheckpointAndRestoreHelper(fileManager.getModelDirAndPrefix(),
                                                   True,
                                                   tf.get_default_graph())

        A = tf.get_default_graph().get_tensor_by_name("A:0")
        B = tf.get_default_graph().get_tensor_by_name("B:0")
        with tf.Session() as sess:
            restoreHelper.restoreFromCheckpoint(sess)
            assert A.eval() == 10
            assert B.eval() == 15
        
    def test_RestoreMetaFail(self, request):
        """
        Tests if the model fails to restore correctly when there is no meta graph to restore from.
        """
        fileManager = FileManager(request.node.name, None)

        with pytest.raises(OSError):
            restoreHelper = CheckpointAndRestoreHelper(fileManager.getModelDirAndPrefix(),
                                                       True,
                                                       tf.get_default_graph())

class Test_FileManager:
    """
    Tests for the FileManager class.
    """
    def test_WithoutRestore(self):
        """
        Tests if the correct model paths are generated if no model to restore from is provided.
        """

        manager = FileManager("TestRegressor")
        TIMESTAMP = datetime.utcnow().strftime("%Y%m%d-%H%M")

        assert manager.getModelDir() == os.getcwd() + "/models/TestRegressor/" +  TIMESTAMP
        assert manager.getModelDirAndPrefix() == os.getcwd() + "/models/TestRegressor/" + TIMESTAMP + "/model"

    def test_WithRestore(self):
        """
        Tests if the correct model paths are generated if a model to restore from is provided.
        """

        RESTORE_FROM = "20171225-1200"
        manager = FileManager("TestRegressor", RESTORE_FROM)

        assert manager.getModelDir() == os.getcwd() + "/models/TestRegressor/" +  RESTORE_FROM
        assert manager.getModelDirAndPrefix() == os.getcwd() + "/models/TestRegressor/" + RESTORE_FROM + "/model"
