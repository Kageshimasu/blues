import cv2


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def show_image(image, pred_bbox, pred_color, teacher_bbox=None, teacher_color=(0, 0, 255), wait_time=1):
    for i in range(pred_bbox.shape[0]):
        bbox = pred_bbox[i]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), pred_color, thickness=2)

    if teacher_bbox is not None:
        for i in range(teacher_bbox.shape[0]):
            bbox = teacher_bbox[i]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), teacher_color, thickness=2)

    cv2.imshow(image)
    cv2.imshow(wait_time)
