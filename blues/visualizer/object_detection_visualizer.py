import cv2
import numpy as np


def show_image_for_object_detection(
        images, pred,
        teacher=None, pred_color=(0, 0, 255), teacher_color=(0, 255, 0), wait_time=1, threshold=0.2, max_detections=50, idx=0):
    pass
    # TODO: object detection用に画像表示
    # image = images[idx].transpose(1, 2, 0)
    # teacher_bbox = teacher[idx]
    # pred_bbox = pred[2][idx]
    #
    # for i in range(pred_bbox.shape[0]):
    #     bbox = pred_bbox[i]
    #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), pred_color, thickness=2)
    #
    # if teacher_bbox is not None:
    #     for i in range(teacher_bbox.shape[0]):
    #         bbox = teacher_bbox[i]
    #         cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), teacher_color, thickness=1)
    #
    # cv2.imshow('training img', image)
    # cv2.waitKey(wait_time)
