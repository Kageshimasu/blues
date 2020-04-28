import cv2
import numpy as np

from ..base.base_dataset import BaseDataset
from ..base.base_data_augmentor import BaseDataAgumentor
from ..common.data import Data


class ObjectDetectionDataset(BaseDataset):

    def __init__(self, inputs: list, teachers: list, batch_size: int,
                 transformers: list = None, augmentors: BaseDataAgumentor = None):
        """
        :param inputs: images path list
        :param teachers: teachers list
        :param batch_size:
        """
        super().__init__(inputs, teachers, batch_size, transformers, augmentors)

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration()
        inputs_path = self._inputs[self._i:self._i + self._batch_size]
        inputs = np.array([cv2.imread(image_path) for image_path in inputs_path])
        teachers = self._teachers[self._i:self._i + self._batch_size]
        file_names = inputs_path
        self._i += self._batch_size

        if self._transformers is not None:
            for transformer in self._transformers:
                inputs, teachers = transformer(inputs, teachers)

        if self._augmentors is not None:
            for augmentor in self._augmentors:
                inputs, teachers = augmentor(inputs, teachers)

        return Data(inputs, teachers, file_names)
