from abc import ABCMeta, abstractmethod
from .base_dataset import BaseDataset


class BaseKFolder(metaclass=ABCMeta):

    def __init__(self, dataset: BaseDataset, n_splits: int = 3):
        self._i = 0
        self._dataset = dataset
        self._dataset_class = dataset.__class__
        self._inputs = dataset.get_inputs()
        self._teachers = dataset.get_teachers()
        self._batch_size = dataset.get_batch_size()
        self._transformers = dataset.get_transformers()
        self._augmentors = dataset.get_augmentors()
        self._n_splits = n_splits
        self._n_samples = len(self._dataset) // n_splits

    @abstractmethod
    def __next__(self):
        pass

    def __iter__(self):
        return self

    def __len__(self):
        return self._n_splits
