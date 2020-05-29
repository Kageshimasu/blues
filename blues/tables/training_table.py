import re
from typing import Dict

from ..base.base_model import BaseModel


class TrainingTable:

    def __init__(self, learning_dir: Dict[str, BaseModel]):
        """
        :param learning_dir:
        {
        'fold1': model1()
        'fold2': model2()
        'fold3': model3()
        }
        """
        if len(learning_dir) == 0:
            raise ValueError('the tables has one value or over')
        for key in learning_dir.keys():
            if not re.match(r'fold[0-9]+', key):
                raise ValueError('the definition of the tables is not appropriate.')
            if key == 'fold0':
                raise ValueError('fold number must be over 0')
        self._learning_dir = learning_dir
        self._i = 0
        self._len = len(learning_dir)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._len:
            self._i = 0
            raise StopIteration()
        self._i += 1
        fold_num = 'fold{}'.format(self._i)
        return fold_num, self._learning_dir[fold_num]
