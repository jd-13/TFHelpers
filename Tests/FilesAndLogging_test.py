"""
Tests for functionality in the FilesAndLogging module.
"""

from datetime import datetime
import os

from TFHelpers.FilesAndLogging import FileManager

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
    