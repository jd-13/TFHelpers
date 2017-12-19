"""
Tests for functionality in the TrainingHelpers module.
"""

import time
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

    def test_CorrectTimeTaken(self):
        """
        ProgressCalculator should provide the time taken since start was called.
        """
        progress = ProgressCalculator(10)
        progress.start()

        time.sleep(1)
        assert progress.timeTaken()[:10] == "0:00:01.00"

        time.sleep(1)
        assert progress.timeTaken()[:10] == "0:00:02.00"

        # Check this still works after a reset
        progress.reset()
        progress.start()

        time.sleep(1)
        assert progress.timeTaken()[:10] == "0:00:01.00"

        time.sleep(1)
        assert progress.timeTaken()[:10] == "0:00:02.00"
