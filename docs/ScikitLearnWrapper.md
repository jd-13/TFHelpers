# Scikit-learn Wrapper

TFHelpers provides a basic wrapper class which you can inherit from to create models which are
compatible with the scikit-learn api. Models which inherit from the `SKTFWrapper` class will be
compatible scikit-learn functionality such as `GridSearchCV`, and will provide `fit` and `predict`
methods for training and producing predictions.

If you inherit from `SKTFWrapper` you will need to build the graph and implement the the training
loop in the `fit` method. The `SciKitWrapper` module will also contain child classes of
`SKTFWrapper` which provide further functionality. Currently only `TFRegressor` is available.

### TFRegressor
If you wish to create a regression model you can inherit for the `TFRegressor` class which provides
a lot of functionality, leaving the methods `_buildGraph` and `_buildModelName` for you to
implement.

`TFRegressor` gives you the following features for free:
* A fairly conventional training loop, with user specified batch size
* Logging of training and validation losses to tensorboard on each epoch, as well as writing any `tf.Summary` nodes at each epoch
* Saving the model at each epoch
* Continuing training from previous runs
* Saving all model and tensorboard related files in separate directories, sorted by model hyperparameters and start time
* Early stopping
* Predicted time to completion
* Warnings if parts of the graph do not appear to be training

In the `_buildGraph` method you will need to build the graph for your model, and set the `_tensors`
data member.

In the `_buildModelName` method you will need to return a string which adequately
describes your model and its hyperparameters. This string is used to group model files.

See `BasicRegressor` for an example.
