import cv2
import numpy as np

from ..base.base_resizer import BaseResizer
from ..utils.object_detection_utils import resize_bbox


class ObjectDetectionResizer(BaseResizer):

    def __init__(self, out_size):
        super().__init__(out_size)

    def __call__(self, inputs, bboxes):
        """
        :param inputs: tensor
        :param bboxes: tensor
        :return:
        """
        max_num_objects = max([bbox.shape[0] for bbox in bboxes])
        batch_size = inputs.shape[0]
        new_inputs = []
        new_bboxes = -1 * np.ones([batch_size, max_num_objects, 5], dtype=np.int)
        for i in range(batch_size):
            w = inputs[i].shape[1]
            h = inputs[i].shape[0]
            new_bbox = resize_bbox(bboxes[i], (w, h), self._out_size)
            new_input = cv2.resize(inputs[i], self._out_size)
            new_bboxes[i, :new_bbox.shape[0]] = new_bbox
            new_inputs.append(new_input)
        return np.array(new_inputs), new_bboxes
