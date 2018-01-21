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

## TrainingValidator