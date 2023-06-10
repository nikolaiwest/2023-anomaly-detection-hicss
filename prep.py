import os
import json
import numpy as np


class ScrewData:
    """This class handles the loading, processing and shuffling of data for screw
    run scenarios.

    Attributes:
        path (str): Path to the JSON files.
        files (list): List of file names in the directory specified by the path.
        len_target (int): Target length of array after padding.
        file_count (int): Total count of files in the directory.
        torques (ndarray): Numpy array holding torque data.
        angles (ndarray): Numpy array holding angle data.
        labels (ndarray): Numpy array holding labels data.
    """

    def __init__(self, path):
        """Initialize the ScrewData object.

        Args:
            path (str): Path to the JSON files.
        """
        self.path = path
        self.files = os.listdir(path)
        # the target length was determined using a histogram of all lengths with the
        # goal to minimize the overall padding of the screw data time series
        self.len_target = 750
        self.file_count = len(self.files)
        # Initialize empty ndarrays for torques, angles and labels.
        self.torques = np.empty((self.file_count, self.len_target))
        self.angles = np.empty((self.file_count, self.len_target))
        self.labels = np.empty(self.file_count)
        # Load and process data from the files.
        self._load_and_process_data()
        # Shuffle the processed data.
        self.torques, self.angles, self.labels = self._shuffle_data(
            self.torques, self.angles, self.labels
        )

    def _load_and_process_data(self) -> None:
        """Loads and processes data from JSON files.

        Iterates through the files, loading them, extracting relevant data, and
        populating the torques, angles, and labels arrays.
        """
        for i, file in enumerate(self.files):
            if i == self.file_count:
                break

            with open(self.path + file, "r") as f:
                screw_run = json.load(f)
            # Extract torque and angle values from the JSON data.
            torque = [
                screw_step["graph"]["torque values"]
                for screw_step in screw_run["tightening steps"]
            ]
            angle = [
                screw_step["graph"]["angle values"]
                for screw_step in screw_run["tightening steps"]
            ]
            # Populate the torque, angle, and label arrays.
            self.torques[i, :] = self._flatten_and_pad(torque, self.len_target)
            self.angles[i, :] = self._flatten_and_pad(angle, self.len_target)
            self.labels[i] = 0 if screw_run["result"] == "OK" else 1

    def _flatten_and_pad(self, data, len_target) -> list:
        """Flattens and pads the input list to a specified target length.

        Args:
            data (list): The list to be flattened and padded.
            len_target (int): The target length of the flattened and padded list.

        Returns:
            ndarray: A flattened and padded version of the input data.
        """
        flattened = [item for sublist in data for item in sublist]
        # If the flattened list is longer than the target length, truncate it.
        if len(flattened) > len_target:
            output = flattened[:len_target]
        # If the flattened list is shorter than the target length, pad it.
        else:
            output = np.pad(
                flattened,
                (0, len_target - len(flattened)),
                "constant",
                constant_values=0,
            )
        return output

    def _shuffle_data(self, *arrays):
        """Shuffles data in the provided arrays.

        This function asserts that all input arrays have the same length, and
        shuffles them.

        Args:
            *arrays: The arrays to be shuffled.

        Returns:
            tuple: A tuple containing the shuffled arrays.
        """
        assert len(set(map(len, arrays))) == 1

        # Seed the RNG for reproducibility.
        np.random.seed(42)
        permutation = np.random.permutation(len(arrays[0]))
        shuffled_arrays = []
        for array in arrays:
            shuffled = np.empty(array.shape, dtype=array.dtype)
            # Shuffle the array according to the generated permutation.
            for old_index, new_index in enumerate(permutation):
                shuffled[new_index] = array[old_index]
            shuffled_arrays.append(shuffled)

        return tuple(shuffled_arrays)

    def get_data(self):
        """Retrieves the torque and label data.

        Returns:
            tuple: A tuple containing the torque and label arrays.
        """
        return self.torques, self.labels
