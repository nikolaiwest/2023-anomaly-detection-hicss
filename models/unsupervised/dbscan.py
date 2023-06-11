from sklearn.cluster import DBSCAN
from numpy import where, logical_not
from utilities import get_metrics


class CustomDBSCAN(DBSCAN):
    """
    Custom DBSCAN model class that adds a fit_and_predict method for convenience.
    Inherits from sklearn's DBSCAN class.

    Attributes:
        Same as sklearn.cluster.DBSCAN
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        """
        Constructor for the CustomDBSCAN class.

        Parameters:
            Same as sklearn.cluster.DBSCAN
        """
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )

    def fit_and_predict(self, X_test, y_test):
        """
        Fits the model and makes predictions on the test data. It also computes the
        evaluation metrics.

        Args:
            X_test (array-like or iterable): The test samples.
            y_test (array-like or iterable): The true labels for test samples.

        Returns:
            dict: A dictionary with evaluation metrics.
        """
        # Fit model
        self.fit(X_test)

        # Apply model to test data
        predictions = self.labels_

        # Mark all cluster other than 0 as 1 (=anomalies)
        predictions = where(predictions != 0, 1, predictions)

        # Evaluate the prediction using classification metrics
        metrics_dict = get_metrics(y_test, predictions)

        # Check for True-/ False-Count consistency
        tp = metrics_dict["TP"]
        tn = metrics_dict["TN"]
        fp = metrics_dict["FP"]
        fn = metrics_dict["FN"]
        # DBSCAN switched True and False
        if (tp + fp) > (tn + fn):  # only applicable for unbalanced scenarios
            # Convert and recalcuate metrics
            predictions = logical_not(predictions).astype(int)
            metrics_dict = get_metrics(y_test, predictions)

        return metrics_dict
