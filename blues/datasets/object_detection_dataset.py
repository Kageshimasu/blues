import cv2
import numpy as np

from ..base.base_dataset import BaseDataset
from ..base.base_data_augmentor import DataAugmentor
from ..base.base_resizer import BaseResizer
from ..common.data import Data


class ObjectDetectionDataset(BaseDataset):

    def __init__(self, inputs: list, teachers: list, batch_size: int,
                 resizer: BaseResizer, transformers: list = None, augmentor: DataAugmentor = None):
        """
        :param inputs:
        :param teachers:
        :param batch_size:
        :param transformers:
        :param augmentor:
        """
        super().__init__(inputs, teachers, batch_size, resizer, transformers, augmentor)

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration()

        inputs_path = self._inputs[self._i:self._i + self._batch_size]
        teachers = np.array(self._teachers[self._i:self._i + self._batch_size])
        inputs, teachers = self._resizer(inputs_path, teachers)
        file_names = inputs_path

        if self._augmentor is not None:
            inputs, teachers = self._augmentor(inputs, teachers)

        if self._transformers is not None:
            for transformer in self._transformers:
                inputs, teachers = transformer(inputs, teachers)

        self._i += self._batch_size
        return Data(inputs, teachers, file_names)
