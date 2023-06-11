from sklearn.neighbors import LocalOutlierFactor
from numpy import where
from utilities import get_metrics


class CustomLocalOutlierFactor(LocalOutlierFactor):
    """
    Custom Local Outlier Factor (LOF) model class that adds a fit_and_predict
    method for convenience. Inherits from sklearn's LocalOutlierFactor class.

    Attributes:
        Same as sklearn.neighbors.LocalOutlierFactor
    """

    def __init__(
        self,
        n_neighbors=20,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination="auto",
        novelty=False,
        n_jobs=None,
    ):
        """
        Constructor for the CustomLOF class.

        Parameters:
            Same as sklearn.neighbors.LocalOutlierFactor
        """
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            novelty=novelty,
            n_jobs=n_jobs,
        )

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
        predictions = self.fit_predict(X_test)

        # Invert the predictions: inliers are labeled 1, outliers are labeled -1.
        predictions = where(predictions == 1, 0, 1)

        # Evaluate the prediction using classification metrics
        metrics_dict = get_metrics(y_test, predictions)

        return metrics_dict
