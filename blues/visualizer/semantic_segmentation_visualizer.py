import cv2
import numpy as np
from typing import Dict, Tuple


class SemanticSegmentationVisualizer:

    def __init__(
            self,
            label_dict: Dict[int, str],
            color_dict: Dict[int, Tuple[int, int, int]],
            transpose_to_numpy: bool = True,
            wait_time: int = 1):
        """
        :param label_dict: [id, class_name]
        :param color_dict: [id, bgr]
        :param transpose_to_numpy: torch or numpy
        :param wait_time:
        """
        self._label_dict = label_dict
        self._color_dict = color_dict
        self._transpose_to_numpy = transpose_to_numpy
        self._wait_time = wait_time

    def __call__(self, inputs: np.ndarray, preds: np.ndarray, teachers: np.ndarray):
        """
        :param inputs: [batch, channel, height, width]
        :param preds:
        :param teachers:
        :return:
        """
        print(self._encode_label(preds[0]))

    def _encode_label(self, labelmap, mode='RGB'):
        labelmap = labelmap.astype('int')
        labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                                dtype=np.uint8)
        for label in self._unique(labelmap):
            if label < 0:
                continue
            labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                np.tile(self._color_dict[label],
                        (labelmap.shape[0], labelmap.shape[1], 1))

        if mode == 'BGR':
            return labelmap_rgb[:, :, ::-1]
        else:
            return labelmap_rgb

    @staticmethod
    def _unique(ar, return_index=False, return_inverse=False, return_counts=False):
        ar = np.asanyarray(ar).flatten()

        optional_indices = return_index or return_inverse
        optional_returns = optional_indices or return_counts

        if ar.size == 0:
            if not optional_returns:
                ret = ar
            else:
                ret = (ar,)
                if return_index:
                    ret += (np.empty(0, np.bool),)
                if return_inverse:
                    ret += (np.empty(0, np.bool),)
                if return_counts:
                    ret += (np.empty(0, np.intp),)
            return ret
        if optional_indices:
            perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
            aux = ar[perm]
        else:
            ar.sort()
            aux = ar
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))

        if not optional_returns:
            ret = aux[flag]
        else:
            ret = (aux[flag],)
            if return_index:
                ret += (perm[flag],)
            if return_inverse:
                iflag = np.cumsum(flag) - 1
                inv_idx = np.empty(ar.shape, dtype=np.intp)
                inv_idx[perm] = iflag
                ret += (inv_idx,)
            if return_counts:
                idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
                ret += (np.diff(idx),)
        return ret