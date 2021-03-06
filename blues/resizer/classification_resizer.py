import cv2
import os
import numpy as np
from typing import List

from ..base.base_resizer import BaseResizer


class ClassificationResizer(BaseResizer):

    def __init__(self, out_size):
        super().__init__(out_size)

    def __call__(self, inputs: List[str], teachers: List):
        if len(inputs) != len(teachers):
            raise ValueError('the size of inputs and that of teachers do not match')
        batch_size = len(inputs)
        ret_inputs = np.zeros((batch_size, self._out_size[0], self._out_size[1], 3), dtype=np.float32)
        for i in range(batch_size):
            img_path = inputs[i]
            if type(img_path) is not str:
                raise ValueError('input type must be string')
            if not os.path.isfile(img_path):
                raise ValueError('{} not found'.format(img_path))
            img = cv2.imread(img_path)
            ret_inputs[i] = cv2.resize(img, self._out_size)
        return ret_inputs, teachers
