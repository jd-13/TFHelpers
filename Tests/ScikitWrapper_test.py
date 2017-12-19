"""
Tests for functionality in the ScikitWrapper module.
"""

import pytest

from TFHelpers.ScikitWrapper import SKTFWrapper

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
        