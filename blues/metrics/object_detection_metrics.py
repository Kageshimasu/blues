import numpy as np


def iou(inputs, teachers):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    teachers: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    ious = []

    scores, boxes, labels = inputs
    if len(boxes.shape) != 3:
        return 0

    for i in range(teachers.shape[0]):
        bbox_annotation = teachers[i, :, :]
        bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

        for cls in np.unique(bbox_annotation[:, 4]).tolist():
            bbox_pred = boxes[i, :, :]
            a = bbox_pred[labels[i] == cls]
            b = bbox_annotation[bbox_annotation[:, 4] == cls]

            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            iw = np.minimum(np.expand_dims(
                a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
            ih = np.minimum(np.expand_dims(
                a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

            iw = np.maximum(iw, 0)
            ih = np.maximum(ih, 0)

            ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                                (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
            ua = np.maximum(ua, np.finfo(float).eps)
            intersection = iw * ih
            iou_nk = intersection / ua
            if len(iou_nk) == 0:
                continue
            iou = sum(sum(iou_nk)) / (iou_nk.shape[0] * iou_nk.shape[1])
            ious.append(iou)
    if len(ious) == 0:
        return 0
    return sum(ious) / len(ious)


def ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
