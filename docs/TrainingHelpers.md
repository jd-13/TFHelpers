# TrainingHelpers

Classes which provide functionality within the training loop of a tensorflow model. To see examples
of how these classes are used, refer to the `fit` method of `TFRegressor`.

## EarlyStoppingHelper
Implements early stopping in your model by checking the loss at each epoch.

    __init__(self, maxChecksWithoutProgress: int)
Construct this object shortly before your training loop, where `maxChecksWithoutProgress` is the
number of epochs to continue without a declining loss before `shouldStop` returns `False`.

    restoreBestModelParams(self) -> bool
Call this after your training loop to restore the model to the state which achived the lowest loss.

    shouldStop(self, lossVal: float) -> bool
Call this every epoch, providing the validation loss in `lossVal`. If `lossVal` is lower than any
previous value then state of the model will be stored. Returns `False` if the number of times this
has been called without lossVal decreasing has exceeded the value `maxChecksWithoutProgress` as
provided in the constructor.

## ProgressCalculator
Records the time taken and estimates the time remaining.

    __init__(self, targetIterations: float)
Construct this object shortly before your training loop, where `targetIterations` is the number of
epochs the model is expected to run for.

    start(self) -> None
Starts the timer. Call this immediately before entering the training loop

    reset(self) -> None
Resets the timer and all counters.

    updateInterval(self, newlyCompletedIterations: float) -> None
Given the number of iterations completed since this was last called (`newlyCompletedIterations`),
updates the expected time to completion.

    getSecondsRemaining(self) -> float
    getTimeStampRemaining(self) -> str
Returns the time remaining.

    timeTaken(self) -> str
Returns the time taken so far.

## TrainingValidator
Provides multiple checks which can be performed while the model is training.

    __init__(self, graph, session)
Construct this object shortly before your training loop, where `graph` and `session` are the graph
and session used for your model.

    validate(self, loss: float) -> None
Run this once on each epoch, where `loss` is the validation loss for the current epoch. You may
wish to call this for only the first few epochs.

This performs the following checks, and raises a `RuntimeWarning` for any checks that fail:

* If any variables in TRAINABLE_VARIABLES haven't changed between epochs, which could indicate parts
of the graph are not being trained due to a programming error
* If the loss value is exactly zero, which could indicate a programming error
* If the standard deviation within a set of trainable variables becomes too high, which could
indicate exploding gradients