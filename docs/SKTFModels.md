# SKTFModels

This module provides examples of models built using the `SciKitWrapper` module. This is a good place
to start if you wish to implement your own models using this wrapper.

## BasicRegressor
Currently this module provides only the `BasicRegressor` class, which inherits from `TFRegressor`
and implements a basic multilayer feedforward network for regressions. Its hyperparameters are to be
passed to the constructor as follows, and provide the following options:

    BasicRegressor(self,
                   learningRate=0.01,
                   batchSize=1000,
                   initializer=tf.contrib.layers.variance_scaling_initializer(),
                   dropoutRate=0.01,
                   restoreFrom=None,
                   hiddenNeuronsList=[10])

* `learningRate`: Provided to the gradient descent algorithm, in this case `tf.train.AdamOptimizer`
* `batchSize`: Number of rows of data to operate on at once
* `initializer`: The initializer to use as the kernel initializer for each layer
* `dropoutRate`: Not yet implemented in this model, does nothing
* `restoreFrom`: If this model has been trained previously with the same hyperparameters, provide
the date time string of the form YYYYMMDD-HHmm which corresponds to the previous training run. The
`CheckpointAndRestoreHelper` will then look this up from the models directory.
* `hiddenNeuronsList`: A list of integers which describes the number of neurons in each hidden layer.
Eg. [100, 50, 30] would mean 100 neurons in the first layer, 50 in the second and 30 in the third
layer.
