# Scikit-learn Wrapper

TFHelpers provides a basic wrapper class which you can inherit from to create models which are
compatible with the scikit-learn api. Models which inherit from the `SKTFWrapper` class will be
compatible with scikit-learn functionality such as `GridSearchCV`, and will provide `fit` and
`predict` methods for training and producing predictions.

If you inherit from `SKTFWrapper` you will need to build the graph and implement the training loop
in the `fit` method. The `SciKitWrapper` module also contains child classes of `SKTFWrapper` which
provide further functionality. Currently only `TFRegressor` is available.

## TFRegressor
If you wish to create a regression model you can inherit from the `TFRegressor` class which provides
a lot of functionality, leaving the methods `_buildGraph` and `_buildModelName` for you to
implement.

Using `TFRegressor` gets you the following features for free:

* A fairly conventional training loop, with user specified batch size
* Logging of training and validation losses to tensorboard on each epoch, as well as writing any `tf.Summary` nodes at each epoch
* Saving the model at each epoch
* Continuing training from previous runs
* Saving all model and tensorboard related files in a directory structure sorted by model hyperparameters and start time
* Early stopping
* Predicted time to completion
* Warnings if issues are detected during training such as: parts of the graph do not appear to be training, the loss is zero, or there is a risk of exploding gradients

### Implementation
    _buildModelName()
In this method you will need to return a string which adequately describes your model and its
hyperparameters. This string is used to group model files.

    _buildGraph()
In this method you will need to build the graph for your model, and return a `RegressorTensors`
object which contains the appropriate operations from your graph. This object is the interface
between the graph which you have created in `_buildGraph` and the training loop implemented in
`TFRegressor`.

    RegressorTensors(self,
                     X_in,
                     y_in,
                     logits,
                     loss,
                     trainingOp,
                     init,
                     saver,
                     dropoutKeepProb)

The constructor of the `RegressorTensors` object takes several operations from your graph that the
training loop will use while training the model. Each is explained below:

* `X_in`: The `tf.Placeholder` for your features
* `y_in`: The `tf.Placeholder` for your labels
* `logits`: The output of your graph. The predict method will call `logits.eval(...)` to produce predictions
* `loss`: The loss of your graph. `loss.eval(...)` will be called to create training and validation loss values to log in tensorboard
* `trainingOp`: The operator that should be evaluated for each batch and epoch to train your model
* `init`: The `tf.global_variables_initializer()` that should be used to initialise your graph
* `saver`: The `tf.train.Saver()` that should be used to save the model
* `dropoutKeepProb`: A `tf.Placeholder` which will be set to the dropout rate during training

For complete examples of both of the above methods, see `SKTFModels.BasicRegressor`.
