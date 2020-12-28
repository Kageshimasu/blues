import numpy as np
import cv2
import torchvision

from typing import List, Callable
from ..base.base_dataset import BaseDataset


class ClassificationDataset(BaseDataset):

    def __init__(self, inputs: List[str], teachers: List[np.ndarray], batch_size: int,
                 transformers: List[Callable] = None, augmentor: torchvision.transforms.Compose = None):
        """
        :param inputs:
        :param teachers:
        :param batch_size:
        :param transformers:
        :param augmentor:
        """
        super().__init__(inputs, teachers, batch_size, transformers, augmentor)

    def __getitem__(self, i):
        input_path = self._inputs[i]
        teacher_data = self._teachers[i]
        input_data = cv2.imread(input_path)

        if self._transformers is not None:
            for transformer in self._transformers:
                input_data, teacher_data = transformer(input_data, teacher_data)

        if self._augmentor is not None:
            input_data = self.transform(image=input_data)["image"]

        return input_data, teacher_data
