# the usual tools
import numpy as np

# tensorflow
import tensorflow as tf

class EarlyStoppingHelper:
    """Contains most of the functionality needed to implement early stopping."""

    MAX_CHECKS_WITHOUT_PROGRESS = 20

    bestLossVal = np.infty
    checksSinceLastProgress = 0
    bestModelParams = None

    def __init__(self, maxChecksWithoutProgress=20):
        self.MAX_CHECKS_WITHOUT_PROGRESS = maxChecksWithoutProgress

    def getModelParams(self):
        """Returns a dictionary of tf.GraphKeys.GLOBAL_VARIABLES"""
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value
                for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

    def restoreBestModelParams(self):
        """If we have the best model parameters saved, this will restore them."""
        success = False

        if self.bestModelParams:
            gvar_names = list(self.bestModelParams.keys())
            assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                          for gvar_name in gvar_names}

            init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
            feed_dict = {init_values[gvar_name]: self.bestModelParams[gvar_name] for gvar_name in gvar_names}
            tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

            success = True

        return success

    def shouldStop(self, lossVal):
        """Returns True if you should stop. Check this at the end of each epoch."""

        if lossVal < self.bestLossVal:
            self.bestLossVal = lossVal
            self.checksSinceLastProgress = 0
            self.bestModelParams = self.getModelParams()
        else:
            self.checksSinceLastProgress += 1

        return self.checksSinceLastProgress > self.MAX_CHECKS_WITHOUT_PROGRESS

