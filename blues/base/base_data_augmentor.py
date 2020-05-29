import imgaug.augmenters as iaa
import numpy as np
from abc import ABCMeta, abstractmethod


class DataAugmentor(metaclass=ABCMeta):

    def __init__(self, sequential: iaa.Sequential):
        """
        :param sequential:
        """
        self._sequential = sequential

    def __len__(self):
        return len(self._sequential)

    @abstractmethod
    def __call__(self, inputs: np.ndarray, teachers: np.ndarray) -> tuple:
        """
        :param inputs:
        :param teachers:
        :return: (inputs, teachers)
        """
        pass
