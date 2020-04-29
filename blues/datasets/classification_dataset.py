import cv2
import numpy as np
import torch

from ..base.base_dataset import BaseDataset
from blues.data_augmentations.data_augmentor import DataAugmentor
from ..common.data import Data


class ClassificationDataset(BaseDataset):

    def __init__(self, inputs: list, teachers: list, batch_size: int,
                 transformers: list = None, augmentors: DataAugmentor = None):
        """
        :param inputs:
        :param teachers:
        :param batch_size:
        :param transformers:
        :param augmentors:
        """
        super().__init__(inputs, teachers, batch_size, transformers, augmentors)

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

        if self._augmentors is not None:
            for augmentor in self._augmentors:
                inputs, teachers = augmentor(inputs, teachers)

        return Data(inputs, teachers, file_names)
