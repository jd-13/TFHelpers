"""
Tests for functionality in the TrainingHelpers module.
"""

import time
import pytest

from TFHelpers.TrainingHelpers import ProgressCalculator

class Test_ProgressCalculator_InvalidBehaviour:
    """
    Tests that the ProgressCalculator behaves appropriately when operated incorrectly.
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

    def test_UpdateBeforeStart(self):
        """
        ProgressCalculator should raise an exception if updateInterval is called before start.
        """
        progress = ProgressCalculator(10)

        with pytest.raises(RuntimeError):
            progress.updateInterval(1)

class Test_ProgressCalculator_Operation:
    """
    Tests that the ProgressCalculator behaves appropriately when under normal operation.
    """

    def test_CorrectTimeTaken(self):
        """
        ProgressCalculator should provide the time taken since start was called.
        """
        progress = ProgressCalculator(10)
        progress.start()

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:01.0"

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:02.0"

        # Check this still works after a reset
        progress.reset()
        progress.start()

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:01.0"

        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:02.0"

        # Check that the interval doesn't have any affect
        progress.updateInterval(1)
        
        time.sleep(1.001)
        assert progress.timeTaken()[:9] == "0:00:03.0"
    
    def test_CorrectTimeRemaining_ConstantSpeed(self):
        """
        ProgressCalculator should provide an estimate of the time remaining for a constant time
        interval.
        """
        progress = ProgressCalculator(3)
        progress.start()

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:02.0"
        assert progress.getSecondsRemaining() == pytest.approx(2, 0.1)

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:01.0"
        assert progress.getSecondsRemaining() == pytest.approx(1, 0.1)

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining() == "0:00:00"
        assert progress.getSecondsRemaining() == pytest.approx(0, 0.1)

    
    def test_CorrectTimeRemaining_VaryingSpeed(self):
        """
        ProgressCalculator should provide an estimate of the time remaining for a varying time
        interval.
        """
        progress = ProgressCalculator(3)
        progress.start()

        time.sleep(1.01)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:02.0"
        assert progress.getSecondsRemaining() == pytest.approx(2, 0.1)

        time.sleep(2.02)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining()[:9] == "0:00:01.5"
        assert progress.getSecondsRemaining() == pytest.approx(1.5, 0.1)

        time.sleep(3.03)
        progress.updateInterval(1)
        assert progress.getTimeStampRemaining() == "0:00:00"
        assert progress.getSecondsRemaining() == pytest.approx(0, 0.1)
