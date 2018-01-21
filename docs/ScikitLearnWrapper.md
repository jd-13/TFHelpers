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
a lot of functionality, leaving the methods `_buildGraph`, `_restoreGraph`, `_buildModelName`, and
the constructor for you to implement.

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
    __init__(self,
            learningRate,
            batchSize,
            initializer,
            dropoutRate,
            restoreFrom,
            outputLength)
The `__init__` method should be overriden and used to set the hyperparameters for your own model,
and should call `TFRegressor.__init__` to provide the hyperparameters required by the `TFRegressor`.

*NOTE: For compatitbility with scikit-learn functionality such as `GridSearchCV`, every parameter
passed to the constructor of your model must be saved to a member variable of exactly the same name.*

    _buildModelName(self)
In this method you will need to return a string which adequately describes your model and its
hyperparameters. This string is used to group model files.

    _buildGraph(self, numFeatures)
In this method you will need to build the graph for your model, and return a `RegressorTensors`
object which contains the appropriate tensors/operations from your graph. This object is the
interface between the graph which you have created in `_buildGraph` and the training loop
implemented in `TFRegressor`. 

The parameter `numFeatures` is the number of columns in the parameter `X` provided to the `fit`
method.

    _restoreGraph(self, graph)
This method must also return a `RegressorTensors` object, however the tensors provided to the
constructor of `RegressorTensors` must be recovered from the provided `graph` using either
`graph.get_tensor_by_name(...)` or `graph.get_operation_by_name(...)`.

    RegressorTensors(self,
                     X_in,
                     y_in,
                     logits,
                     loss,
                     trainingOp,
                     dropoutKeepProb)

The constructor of the `RegressorTensors` object takes several operations from your graph that the
training loop will use while training the model. Each is explained below:

* `X_in`: The `tf.Placeholder` for your features
* `y_in`: The `tf.Placeholder` for your labels
* `logits`: The output of your graph. The predict method will call `logits.eval(...)` to produce predictions
* `loss`: The loss of your graph. `loss.eval(...)` will be called to create training and validation loss values to log in tensorboard
* `trainingOp`: The operator that should be evaluated for each batch and epoch to train your model
* `dropoutKeepProb`: A `tf.Placeholder` which will be set to the dropout rate during training

For complete examples of all of the above methods, see `SKTFModels.BasicRegressor`.

### Usage
The interface to train models which inherit from `TFRegressor` is the `fit` and `predict` methods.

    fit(self, X, y, X_valid, y_valid, numEpochs)
This will call either `_buildGraph` or `_restoreGraph` and then train the model for `numEpochs`,
using `X` as the features and `y` as the labels. A validation set must also be provided in `X_valid`
and `y_valid`.

    predict(self, X)
Will perform inference on the given dataset `X`, returning a numpy array of predictions. Handles an
`X` that has a large number of parameters by splitting it into batches the same size as specified in
the constructor.