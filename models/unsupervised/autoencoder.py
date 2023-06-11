from keras import Model, layers, Sequential


class CustomAutoencoder(Model):
    """Custom Autoencoder model that inherits from Keras Model.

    This class implements an Autoencoder, which is a type of neural network
    used for learning efficient codings of input data. It's composed of
    an encoder, which learns the data representation, and a decoder, which
    tries to reconstruct the input data from its learned representation.
    """

    def __init__(self, len_target):
        """Initialize the Autoencoder model.

        Args:
            len_target (int): The desired output size (dimensionality of output space).
        """
        super(CustomAutoencoder, self).__init__()

        # Define the encoder part of the autoencoder
        self.encoder = Sequential(
            [
                # First layer with 128 units and ReLU activation
                layers.Dense(128, activation="relu"),
                # Second layer with 64 units and ReLU activation
                layers.Dense(64, activation="relu"),
                # Third layer with 32 units and ReLU activation
                layers.Dense(32, activation="relu"),
            ]
        )

        # Define the decoder part of the autoencoder
        self.decoder = Sequential(
            [
                # First layer with 64 units and ReLU activation
                layers.Dense(64, activation="relu"),
                # Second layer with 128 units and ReLU activation
                layers.Dense(128, activation="relu"),
                # Third layer with len_target units and sigmoid activation
                layers.Dense(len_target, activation="sigmoid"),
            ]
        )

    def call(self, inputs):
        """Perform the forward pass in the autoencoder.

        This function is called during the forward pass of the model.
        It transforms the input data using the encoder, and then reconstructs
        the original input using the decoder.

        Args:
            inputs (Tensor): The input data to be passed through the autoencoder.

        Returns:
            Tensor: The reconstructed output from the autoencoder.
        """
        # Encode the inputs
        encoded = self.encoder(inputs)

        # Decode the encoded representation
        decoded = self.decoder(encoded)

        # Return the decoded (reconstructed) representation
        return decoded
