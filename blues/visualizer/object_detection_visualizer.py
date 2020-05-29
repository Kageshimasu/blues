import cv2
import numpy as np
from typing import Dict, Tuple


class ObjectDetectionVisualizer:

    def __init__(
            self,
            label_dict: Dict[int, str],
            pred_color: Tuple[int, int, int] = (0, 0, 255),
            teacher_color: Tuple[int, int, int] = (0, 255, 0),
            index_to_show: int = 0,
            transpose_to_numpy: bool = True,
            wait_time: int = 1):
        """
        :param label_dict:
        :param pred_color:
        :param teacher_color:
        :param index_to_show:
        :param transpose_to_numpy:
        :param wait_time:
        """
        self._label_dict = label_dict
        self._pred_color = pred_color
        self._teacher_color = teacher_color
        self._index_to_show = index_to_show
        self._transpose_to_numpy = transpose_to_numpy
        self._wait_time = wait_time

    def __call__(self, inputs, preds, teachers):
        img = inputs[self._index_to_show]
        if self._transpose_to_numpy:
            img = img.transpose(1, 2, 0)
        pred = preds[self._index_to_show]

        for class_id in range(len(pred)):
            bbox = pred[class_id]
            if bbox is None:
                continue
            if bbox.shape[0] < 0:
                continue
            detections = bbox.shape[0]
            for n_obj in range(detections):
                if class_id < 0:
                    continue
                x_min, y_min, x_max, y_max = \
                    int(bbox[n_obj][0]), int(bbox[n_obj][1]), int(bbox[n_obj][2]), int(bbox[n_obj][3])
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), self._pred_color, thickness=1)
                img = self._draw_text_with_background(
                    img,
                    self._label_dict[class_id],
                    (x_min, y_min),
                    self._pred_color
                )

        if teachers is not None:
            teacher = teachers[self._index_to_show]
            for i in range(teacher.shape[0]):
                bbox = teacher[i]
                class_id = int(bbox[4])
                if class_id < 0:
                    continue
                x_min, y_min, x_max, y_max = \
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), self._teacher_color, thickness=1)
                img = self._draw_text_with_background(
                    img,
                    self._label_dict[class_id],
                    (x_min, y_min),
                    self._teacher_color
                )

        cv2.imshow('training image', img)
        cv2.waitKey(self._wait_time)

    @staticmethod
    def _draw_text_with_background(
            origin_img: np.ndarray,
            text: str,
            offsets: Tuple[int, int],
            background_color: Tuple[int, int, int],
            text_color=(0, 0, 0),
            margin_px=5,
            font_scale=0.5,
            alpha=0.6):
        img = origin_img.copy()
        overlay = img.copy()

        font = cv2.FONT_HERSHEY_DUPLEX
        text_width, text_height = cv2.getTextSize(text, font, font_scale, 1)[0]
        background_coors = (offsets,
                            (int(offsets[0] + text_width + margin_px * 2), int(offsets[1] - text_height - margin_px * 2)))
        img = cv2.rectangle(img, background_coors[0], background_coors[1], background_color, cv2.FILLED)
        img = cv2.putText(img, text, (offsets[0] + margin_px, offsets[1] - margin_px), font, font_scale, text_color, 1,
                          cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
