# TFHelpers

TFHelpers is a small collection of classes which implement common tasks in Tensorflow. The
functionality provided by each module is described in this documentation.

The modules `FilesAndLogging` and `TrainingHelpers` provide functionality such as managing model
files and early stopping, and may be useful in many tensorflow models. The module `ScikitWrapper`
provides several classes which aid with implementing tensorflow models behind a scikit-learn style
API.

If you use the `FilesAndLogging` module you'll also be able to use the `ModelManager` web interface
to sort your models by hyperparameters and generate the appropriate tensorboard commands to compare
different models and training runs.