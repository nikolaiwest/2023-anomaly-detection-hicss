from sklearn.ensemble import IsolationForest
from utilities import get_metrics


class CustomIsolationForest(IsolationForest):
    """
    Custom Isolation Forest model class that modifies the predict method and
    adds a fit_and_predict method for convenience. Inherits from sklearn's
    IsolationForest class.

    Attributes:
        Same as sklearn.ensemble.IsolationForest
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        """
        Constructor for the CustomIsolationForest class.

        Parameters:
            Same as sklearn.ensemble.IsolationForest
        """
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
        )

    def predict(self, X):
        """
        Overridden predict method to switch the labels from -1/1 to 0/1.

        Args:
            X (array-like or iterable): The input samples.

        Returns:
            array, shape (n_samples,): 0 for normal data and 1 for anomalies.
        """
        predictions = super().predict(X)
        predictions[predictions == 1] = 0
        predictions[predictions == -1] = 1
        return predictions

    def fit_and_predict(self, X_test, y_test):
        """
        Fits the model and makes predictions on the test data. It also computes
        the evaluation metrics.

        Args:
            X_test (array-like or iterable): The test samples.
            y_test (array-like or iterable): The true labels for test samples.

        Returns:
            dict: A dictionary with evaluation metrics.
        """
        # Fit model
        self.fit(X_test)

        # Apply model to test data
        predictions = self.predict(X_test)

        # Evaluate the prediction using classification metrics
        metrics_dict = get_metrics(y_test, predictions)

        return metrics_dict
