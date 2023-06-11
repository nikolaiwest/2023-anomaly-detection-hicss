from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

from utilities import TqdmProgressCallback


class CustomLSTM:
    """
    A custom LSTM model class.

    Attributes:
        model (Sequential): The LSTM model.
        history (History): History of model training.

    Methods:
        fit: Trains the LSTM model.
        predict: Makes predictions based on the trained model.
        evaluate: Evaluates the trained model.
    """

    def __init__(self, input_shape, dropout=0.1):
        """
        The constructor for CustomLSTM class.

        Parameters:
            input_shape (tuple): Shape of the input data.
            units (int): Number of LSTM units. Default is 64.
            dropout (float): Dropout rate. Default is 0.2.
        """
        # Initialize the Sequential model
        self.model = Sequential()

        # Add LSTM layers with dropout
        self.model.add(
            LSTM(64, dropout=dropout, return_sequences=True, input_shape=input_shape)
        )
        # self.model.add(LSTM(64, dropout=dropout, return_sequences=True))
        self.model.add(LSTM(32, dropout=dropout))

        # Add a Dense output layer with sigmoid activation
        self.model.add(Dense(1, activation="sigmoid"))

        # Compile the model
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def fit(self, X_train, y_train, epochs=100, batch_size=64, class_weight=None):
        """
        Trains the LSTM model.

        Parameters:
            X_train (array): Training data.
            y_train (array): Training labels.
            epochs (int): Number of epochs. Default is 10.
            batch_size (int): Batch size. Default is 32.
            class_weight (dict): Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function (during
            training only).
        """
        # Set up callbacks
        early_stopping = EarlyStopping(
            monitor="loss", mode="min", verbose=1, patience=100
        )
        progress_bar = TqdmProgressCallback()

        # Fit the model
        self.history = self.model.fit(
            X_train,
            y_train,
            verbose=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, progress_bar],
            class_weight=class_weight,
        )

    def predict(self, X_test):
        """
        Makes predictions based on the trained model.

        Parameters:
            X_test (array): Test data.

        Returns:
            array: Predictions.
        """
        return (self.model.predict(X_test) > 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model.

        Parameters:
            X_test (array): Test data.
            y_test (array): Test labels.

        Returns:
            float: Test loss.
            float: Test accuracy.
        """
        return self.model.evaluate(X_test, y_test)
