import cv2
import os
import numpy as np
from typing import List

from ..base.base_resizer import BaseResizer


class ClassificationResizer(BaseResizer):

    def __init__(self, out_size):
        super().__init__(out_size)

    def __call__(self, input_data: np.ndarray, teacher_data: np.ndarray):
        return cv2.resize(input_data, self._out_size), teacher_data
