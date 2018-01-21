# FilesAndLogging

Several classes which assist with managing model files and creating tensorboard logs. To see
examples of how these classes are used, refer to the `fit` method of `TFRegressor`.


## CheckpointAndRestoreHelper
This class will create a tf.train.Saver(), either as new or by using `import_meta_graph`. It then
allows restoration of an existing graph and/or the saving of a new graph.

    __init__(self, modelRootPath: str, shouldRestore: bool, graph)
Construct this object shortly before your training loop. This creates a saver which if `shouldRestore` is set to `True` will attempt to import the meta graph from existing model files, else will create new files when `saveCheckpoint` is called.

The `modelRootPath` is the path which will be used to save or restore from model files, and should
be the complete path to the files including their prefix. For example, if your model files are
`models/model.ckpt.meta` and `models/model.ckpt.index`, you should provide `models/model`.

After constructing the object, you will be able to retrive tensors and operations from the graph as
normal using `get_tensor_by_name` and `get_operation_by_name`.

    restoreFromCheckpoint(self, sess) -> int
Will restore the model from the latest existing checkpoint.

    saveCheckpoint(self, sess, epoch: int) -> None
Saves the current state of the model and writes a file containing the `epoch` value.

## FileManager
Generates paths in which to save model files in a standardised and consistent way.

    __init__(self, modelName: str, restoreFrom: str)
Generates the path to be used to stores models in, using the current datetime if `restoreFrom` is
`None`, or using the string provided in `restoreFrom`. If restore from is provided, it must be of
the format `YYYYMMDD-HHMM`.

    getModelDir(self) -> str
Returns the complete path to the directory which is to be used to store model files.

    getModelDirAndPrefix(self) -> str
Returns the complete path to the directory which is to be used to store model files, plus the
"model" prefix for the files themselves.

## TensorboardLogHelper
Will write tensorboard logs in the provided directory for all `tf.Summary` nodes in the graph (such
as `tf.summary.histogram`) and also provides the ability to create additional scalar summaries, such
as for training loss and validation loss.

    __init__(self, logDir, graph, summaryNames: List[str])
Construct this object shortly before your training loop. This creates a `tf.summary.FileWriter` for
the given `logDir` and `graph`. If you'd like additional scalar summaries that can be manipluated
during the training loop, provide a list of their names in `summaryNames`

    writeSummary(self, sess, summaryValues: List[float]) -> None
Writes all summaries in the graph.

If you provided anything other than an empty list to the constructor in `summaryNames`, you must
provide a list of the same length in `summaryValues`, containing the values you wish to be written
to the summaries. The values must be provided in the same order as in `summaryNames`.

    close(self) -> None
Closes the `tf.summary.FileWriter`. This should be called only before the object will be destroyed.