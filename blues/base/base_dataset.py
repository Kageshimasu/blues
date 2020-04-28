from abc import ABCMeta, abstractmethod
from ..common.data import Data
from .base_data_augmentor import BaseDataAgumentor


class BaseDataset(metaclass=ABCMeta):

    def __init__(self, inputs: list, teachers: list, batch_size: int,
                 transformers: list = [], augmentors: BaseDataAgumentor = None):
        if len(inputs) != len(teachers):
            raise ValueError('the number of the inputs and that of the teachers did not match')
        if len(inputs) < batch_size:
            raise ValueError('the batch size is too large for the number of the inputs')
        self._inputs = inputs
        self._teachers = teachers
        self._batch_size = batch_size
        self._transformers = transformers
        self._augmentors = augmentors
        self._i = 0

    @abstractmethod
    def __next__(self) -> Data:
        pass

    def __len__(self):
        return len(self._inputs)

    def __iter__(self):
        return self

    def get_inputs(self):
        return self._inputs

    def get_teachers(self):
        return self._teachers

    def get_batch_size(self):
        return self._batch_size

    def get_transformers(self):
        return self._transformers

    def get_augmentors(self):
        return self._augmentors
