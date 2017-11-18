# scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin

class SKTFWrapper(BaseEstimator, RegressorMixin):
    """Doesn't actually do anything, just provides some common functionality used for wrapping TF
    models in an sklearn API"""

    _session = None

    def fit(self, X, y, X_valid, y_valid):
        """Build and train the graph here"""
        pass

    def predict(self, X):
        """Return predictions here"""
        pass

    def _closeSession(self):
        """Ends the tensorflow session if one is open"""
        if self._session:
            self._session.close()
