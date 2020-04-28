from ..base.base_model import BaseModel


class PredictingTable:

    def __init__(self, predicting_table: dir):
        """
        :param predicting_table:
        {
        model1(): acc,
        model2(): acc,
        }
        """
        if len(predicting_table) == 0:
            raise ValueError('the tables has one value or over')
        for key in predicting_table.keys():
            if not isinstance(key, BaseModel):
                raise ValueError('the keys must be BaseModel type')
            if type(predicting_table[key]) is not float:
                raise ValueError('the values must be float type')
        self._learning_dir = predicting_table
        self._keys = list(self._learning_dir.keys())
        self._i = 0
        self._len = len(predicting_table)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._len:
            self._i = 0
            raise StopIteration()
        model = self._keys[self._i]
        acc = self._learning_dir[model]
        self._i += 1
        return acc, model
