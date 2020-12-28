import numpy as np
from sklearn.metrics import accuracy_score, label_ranking_average_precision_score


def accuracy(teachers, preds):
    return accuracy_score(teachers, np.argmax(preds, axis=1))


def label_ranking_average_precision(teachers, preds):
    return label_ranking_average_precision_score(teachers, preds)
