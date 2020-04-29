import random
import numpy as np


class DataAugmentor:

    def __init__(self, data_augmentations: list):
        """
        :param data_augmentations:
        """
        self._data_augmentations = data_augmentations

    def __len__(self):
        return len(self._data_augmentations)

    def __call__(self, inputs: np.ndarray, teachers: np.ndarray):
        """
        :param inputs:
        :param teachers:
        :return:
        """
        prob = len(self) + 1
        for augment_function in self._data_augmentations:
            if random.random() < prob:
                inputs, teachers = augment_function(inputs, teachers)
        return inputs, teachers