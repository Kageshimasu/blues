import cv2
import numpy as np

from ..base.base_resizer import BaseResizer


class ClassificationResizer(BaseResizer):

    def __init__(self, out_size):
        super().__init__(out_size)

    def __call__(self, inputs, teachers):
        batch_size = inputs.shape[0]
        ret_inputs = np.zeros((batch_size, self._out_size[0], self._out_size[1], 3))
        for i in range(batch_size):
            img = inputs[i]
            ret_inputs[i] = cv2.resize(img, self._out_size)
        return ret_inputs, teachers
