import cv2
import numpy as np


def show_image_for_classification(
        images, pred,
        teacher=None, pred_color=(0, 0, 255), teacher_color=(0, 255, 0), wait_time=1, idx=0):
    image = images[idx]
    cv2.imshow('training img', image)
    cv2.waitKey(wait_time)
