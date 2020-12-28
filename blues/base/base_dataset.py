import numpy as np
import torch
import torchvision

from typing import List, Callable
from abc import abstractmethod
from ..base.base_data_augmentor import DataAugmentor


class BaseDataset(torch.utils.data.Dataset):

    def __init__(
            self, inputs: List[str],
            teachers: List[np.ndarray or str],
            batch_size: int,
            transformers: List[Callable],
            augmentor: torchvision.transforms.Compose):
        """
        :param inputs: image path list
        :param teachers: ndarray list
        :param batch_size: batch size
        :param transformers: function list to transform the inputs and teachers
        :param augmentor:
        """
        if len(inputs) != len(teachers):
            raise ValueError('the number of the inputs and that of the teachers did not match,'
                             'length of inputs is {}, length of teachers is {}'.format(len(inputs), len(teachers)))
        self._inputs = inputs
        self._teachers = teachers
        self._batch_size = batch_size
        self._transformers = transformers
        self._augmentor = augmentor

    @abstractmethod
    def __getitem__(self, i) -> np.ndarray:
        pass

    def __len__(self):
        return len(self._inputs)

    def get_batch_size(self):
        return self._batch_size

    def get_inputs(self):
        return self._inputs

    def get_teachers(self):
        return self._teachers

    def get_transformers(self):
        return self._transformers

    def get_augmentor(self):
        return self._augmentor
