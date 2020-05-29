import numpy as np
from typing import List, Callable
from abc import ABCMeta, abstractmethod
from ..common.data import Data
from .base_data_augmentor import DataAugmentor
from .base_resizer import BaseResizer


class BaseDataset(metaclass=ABCMeta):

    def __init__(self, inputs: List[str], teachers: List[np.ndarray or str], batch_size: int,
                 resizer: BaseResizer = None, transformers: List[Callable] = [], augmentor: DataAugmentor = None):
        """
        :param inputs: image path list
        :param teachers: ndarray list
        :param batch_size: batch size
        :param transformers: function list to transform the inputs and teachers
        :param augmentor:
        """
        if len(inputs) != len(teachers):
            raise ValueError('the number of the inputs and that of the teachers did not match')
        if len(inputs) < batch_size:
            raise ValueError('the batch size is too large for the number of the inputs')
        self._inputs = inputs
        self._teachers = teachers
        self._batch_size = batch_size
        self._resizer = resizer
        self._transformers = transformers
        self._augmentor = augmentor
        self._i = 0

    @abstractmethod
    def __next__(self) -> Data:
        pass

    def __len__(self):
        return len(self._inputs)

    def __iter__(self):
        return self

    def get_inputs(self) -> List[str]:
        return self._inputs

    def get_teachers(self) -> List[np.ndarray or str]:
        return self._teachers

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_resizer(self) -> BaseResizer:
        return self._resizer

    def get_transformers(self) -> List[Callable]:
        return self._transformers

    def get_augmentor(self) -> DataAugmentor:
        return self._augmentor
