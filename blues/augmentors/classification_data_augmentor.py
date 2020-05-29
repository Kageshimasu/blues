import imgaug.augmenters as iaa
import numpy as np

from ..base.base_data_augmentor import DataAugmentor


class ClassificationDataAugmentor(DataAugmentor):

    def __init__(self, sequential: iaa.Sequential):
        """
        :param sequential:
        """
        super().__init__(sequential)

    def __call__(self, inputs: np.ndarray, teachers: np.ndarray or str) -> tuple:
        """
        :param inputs:
        :param teachers:
        :return: (inputs, teachers)
        """
        if len(inputs.shape) != 4:
            raise ValueError('inputs shape length must be 4 but got {}'.format(inputs.shape))
        inputs = self._sequential(images=inputs)
        return inputs, teachers
