import cv2
import numpy as np


def show_image(
        images, pred,
        teacher=None, pred_color=(0, 0, 255), teacher_color=(0, 255, 0), wait_time=1, idx=0):
    image = images[idx].transpose(1, 2, 0)
    teacher_id = teacher[idx]
    pred_id = pred[2][idx]

    cv2.imshow('training img', image)
    cv2.waitKey(wait_time)
