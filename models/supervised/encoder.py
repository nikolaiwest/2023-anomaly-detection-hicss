from sktime_dl.classifiers.deeplearning import EncoderClassifier


class CustomEncoder(EncoderClassifier):
    """
    Custom Encoder Classifier inheriting EncoderClassifier class from sktime-dl
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor method for initializing CustomEncoderClassifier object.

        Parameters:
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.
        """
        super().__init__(
            *args, **kwargs
        )  # call the super class (EncoderClassifier) constructor
