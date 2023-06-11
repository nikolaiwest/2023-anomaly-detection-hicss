import matplotlib.pyplot as plt
from numpy import concatenate, unique
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from keras.callbacks import Callback
from datetime import datetime as dt
from tqdm.notebook import tqdm


class TqdmProgressCallback(Callback):
    """Custom callback for use with Keras to display tqdm progress bar during training.

    This class inherits from Keras Callback and uses the tqdm library to display a
    progress bar with the number of epochs completed during the training of a model.
    """

    def on_train_begin(self, logs=None):
        """Initializes the tqdm progress bar at the start of training.

        Args:
            logs (dict, optional): The logs dict contains the loss and metric values
                for the epoch and is intended for use by other callbacks.
        """
        # Retrieve total number of epochs for progress bar
        self.epochs = self.params["epochs"]

        # Initialize the progress bar with total epochs and appropriate descriptors
        self.progress_bar = tqdm(total=self.epochs, desc="Training", unit="epoch")

    def on_epoch_end(self, epoch, logs=None):
        """Updates the tqdm progress bar after each completed epoch.

        Args:
            epoch (int): The index of the epoch just completed.
            logs (dict, optional): The logs dict contains the loss and metric values
                for the epoch and is intended for use by other callbacks.
        """
        # Update the progress bar by one step (i.e., one epoch)
        self.progress_bar.update()

    def on_train_end(self, logs=None):
        """Closes the tqdm progress bar after training is completed.

        Args:
            logs (dict, optional): The logs dict contains the loss and metric values
                for the epoch and is intended for use by other callbacks.
        """
        # Close the progress bar at the end of training
        self.progress_bar.close()


def run_cross_validation(data, labels, k=10) -> None:
    """
    Splits the given data and labels into k subsets for cross validation.

    Args:
        data (numpy.ndarray): The data to split.
        labels (numpy.ndarray): The corresponding labels for the data.
        k (int, optional): The number of subsets to split the data into. Default is 10.

    Yields:
        tuple: A tuple containing fold number, training data, training labels,
        validation data, and validation labels.
    """

    # calculate the size of each fold
    fold_size = len(data) // k

    # iterate over each fold
    for i in range(k):
        # prepare the training data and labels
        train_data = concatenate(
            (data[: i * fold_size], data[(i + 1) * fold_size :]), axis=0
        )
        train_labels = concatenate(
            (labels[: i * fold_size], labels[(i + 1) * fold_size :]), axis=0
        )

        # prepare the validation data and labels
        val_data = data[i * fold_size : (i + 1) * fold_size]
        val_labels = labels[i * fold_size : (i + 1) * fold_size]

        # yield the fold number along with the training and validation data and labels
        yield i + 1, train_data, train_labels, val_data, val_labels


def describe_data(x_train, y_train, x_test, y_test) -> None:
    """
    Prints information about the training and test datasets.

    Args:
        x_train, y_train (numpy.ndarray): The training data and corresponding labels.
        x_test, y_test (numpy.ndarray): The test data and corresponding labels.
    """

    # get unique elements and their counts for training and test labels
    descr_train = unique(y_train, return_counts=True)
    descr_test = unique(y_test, return_counts=True)

    # print size of training and test data
    print(f"Training data:\tx={x_train.shape},\ty={y_train.shape}")

    # print counts of each unique label in the training data
    print(f"\t\t#OK={descr_train[1].astype(int)[0]}")
    print(f"\t\t#NOK={descr_train[1].astype(int)[1]}")

    # print size of test data
    print(f"Test data:\tx={x_test.shape},\ty={y_test.shape}")

    # print counts of each unique label in the test data
    print(f"\t\t#OK={descr_test[1].astype(int)[0]}")
    print(f"\t\t#NOK={descr_test[1].astype(int)[1]}")


def plot_hist(hist) -> None:
    """
    Plots the training and validation loss history.

    Args:
        hist (keras.callbacks.History): History object from model training containing
        loss over epochs.
    """

    # plot the loss for the training and validation data
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])

    # add a title and labels for the y and x axes
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    # add a legend to explain which line is for training and which is for validation
    plt.legend(["Train", "Val"], loc="upper right")

    # display the plot
    plt.show()


def get_metrics(y_true, y_pred) -> dict:
    """
    Calculates various performance metrics for binary classification.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.

    Returns:
        dict: A dictionary containing the calculated metrics (Confusion matrix
        components, Accuracy, Precision, and F1-score).
    """

    # All calculations for the classification metrics are done with the respective
    # sklearn methods, since this may help to prevent potential errors and to improve
    # the overall performance. However, the original calculations are still included
    # in the following section, for reference.

    # confusion matrix elements
    # tp = ((y_pred == 1) & (y_true == 1)).sum()
    # tn = ((y_pred == 0) & (y_true == 0)).sum()
    # fp = ((y_pred == 1) & (y_true == 0)).sum()
    # fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # accuracy
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    accuracy = accuracy_score(y_true, y_pred)

    # precision
    # precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    precision = precision_score(y_true, y_pred)

    # recall (also known as Sensitivity)
    # recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    recall = recall_score(y_true, y_pred)

    # F1 Score
    # f1 = (
    #     2 * (precision * recall) / (precision + recall)
    #     if (precision + recall) != 0
    #     else 0)
    f1 = f1_score(y_true, y_pred)

    # Macro F1 Score
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    # return metrics as a dictionary
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Macro F1 Score": macro_f1,
    }


def print_status(text: str) -> None:
    """
    Prints a given text with the current timestamp.

    The function prefixes the input text with the current timestamp
    in the format 'HH:MM:SS' to give a real-time status update.

    Args:
        text (str): The text to print along with the current timestamp.

    Returns:
        None
    """
    print(f"{dt.now().strftime('%H:%M:%S')}: {text}")


# Get list to specify the column order for all result df
columns_order = [
    "Fold",
    "TP",
    "TN",
    "FP",
    "FN",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]
