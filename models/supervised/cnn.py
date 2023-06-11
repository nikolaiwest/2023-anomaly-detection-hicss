from keras.layers import Dropout, Dense
from keras.models import Model
from sktime_dl.classifiers.deeplearning import CNNClassifier


class CustomCNN(CNNClassifier):
    """Custom CNN Classifier inheriting from CNNClassifier.

    This class adds a dropout layer after the CNN layers to reduce overfitting
    and uses a different loss function, optimizer, and metrics.

    :param dropout_rate: float, the rate of dropout in the Dropout layer
    :param loss: string, the loss function to be used in model training
    :param optimizer: string, the optimizer to be used in model training
    :param metrics: list of string, the metrics to be monitored during training
    """

    def __init__(
        self,
        dropout_rate=0.5,
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
        *args,
        **kwargs
    ):
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        super().__init__(*args, **kwargs)

    def build_model(self, input_shape, nb_classes, **kwargs):
        """Overrides the build_model method of the superclass.

        This method constructs a compiled, un-trained, keras model
        that is ready for training with added dropout layer,
        and with a custom loss function, optimizer and metrics.

        :param input_shape: tuple, shape of the input data
        :param nb_classes: int, number of output classes

        :return: compiled keras model ready for training
        """

        # Building network using superclass method
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        # Adding a Dropout layer to the existing network
        output_layer = Dropout(self.dropout_rate)(output_layer)

        # Adding a dense output layer
        output_layer = Dense(units=nb_classes, activation="sigmoid")(output_layer)

        # Building the model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compiling the model with specified loss function, optimizer, and metrics
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return model
