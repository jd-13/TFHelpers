"""
Tests for functionality in the TrainingHelpers module.
"""

import pytest

from TFHelpers.TrainingHelpers import ProgressCalculator

class Test_ProgressCalculator:
    """
    Tests for the ProgressCalculator class.
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
