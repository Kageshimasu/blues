from abc import ABCMeta, abstractmethod
from ..common.data import Data
from blues.data_augmentations.data_augmentor import DataAugmentor


class BaseDataset(metaclass=ABCMeta):

    def __init__(self, inputs: list, teachers: list, batch_size: int,
                 transformers: list = [], augmentors: DataAugmentor = None):
        """
        :param inputs: image path list
        :param teachers: ndarray list
        :param batch_size: batch size
        :param transformers: function list to transform the inputs and teachers
        :param augmentors:
        """
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

    def get_inputs(self) -> list:
        return self._inputs

    def get_teachers(self) -> list:
        return self._teachers

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_transformers(self) -> list:
        return self._transformers

    def get_augmentors(self) -> DataAugmentor:
        return self._augmentors
