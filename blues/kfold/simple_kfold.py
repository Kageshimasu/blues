from ..base.base_dataset import BaseDataset
from ..base.base_kfolder import BaseKFolder


class KFolder(BaseKFolder):

    def __init__(self, dataset: BaseDataset, n_splits: int = 3):
        """
        KFolder is a folder which assumes that all the data was sampled from the same distribution
        :param dataset:
        :param n_splits:
        """
        super().__init__(dataset, n_splits)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_splits <= self._k:
            raise StopIteration()

        train_inputs = [
            inputs for i, inputs in enumerate(self._inputs)
            if i not in range(self._i, self._n_samples)]
        train_teachers = [
            teachers for i, teachers in enumerate(self._teachers)
            if i not in range(self._i, self._n_samples)]
        test_inputs = self._inputs[self._i:self._i + self._n_samples]
        test_teachers = self._teachers[self._i:self._i + self._n_samples]

        train_dataset = self._dataset_class(
            train_inputs, train_teachers, self._batch_size, self._resizer, self._transformers, self._augmentor)
        valid_dataset = self._dataset_class(
            test_inputs, test_teachers, self._batch_size, self._resizer, self._transformers, None)

        self._i += self._n_samples
        self._k += 1
        return train_dataset, valid_dataset
