from abc import ABCMeta, abstractmethod
from .base_dataset import BaseDataset


class BaseKFolder(metaclass=ABCMeta):

    def __init__(self, dataset: BaseDataset, n_splits: int = 3):
        """
        :param dataset: base dataset
        :param n_splits:
        """
        if n_splits <= 1:
            raise ValueError('The number of splits must be more than 1')
        self._i = 0
        self._k = 0
        self._dataset = dataset
        self._dataset_class = dataset.__class__
        self._inputs = dataset.get_inputs()
        self._teachers = dataset.get_teachers()
        self._batch_size = dataset.get_batch_size()
        self._resizer = dataset.get_resizer()
        self._transformers = dataset.get_transformers()
        self._augmentor = dataset.get_augmentor()
        self._n_splits = n_splits
        self._n_samples = len(self._dataset) // n_splits

    @abstractmethod
    def __next__(self) -> tuple:
        """
        :return: (train_dataset, valid_dataset)
        """
        pass

    def __iter__(self):
        return self

    def __len__(self):
        return self._n_splits
