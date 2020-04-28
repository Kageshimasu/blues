import cv2
import numpy as np

from ..base.base_dataset import BaseDataset
from ..common.data import Data


class PredictionDataset(BaseDataset):

    def __init__(self, inputs: list, batch_size: int, transformers: list = None):
        """
        :param inputs: images path list
        :param batch_size:
        """
        super().__init__(inputs, [None for _ in range(len(inputs))], batch_size, transformers, None)

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration()
        inputs_path = self._inputs[self._i:self._i + self._batch_size]
        inputs = np.array([cv2.imread(image_path) for image_path in inputs_path])
        teachers = np.array(self._teachers[self._i:self._i + self._batch_size])
        file_names = inputs_path
        self._i += self._batch_size

        if self._transformers is not None:
            for transformer in self._transformers:
                inputs, teachers = transformer(inputs, teachers)

        return Data(inputs, teachers, file_names)
