import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(teachers, preds):
    return accuracy_score(teachers, np.argmax(preds, axis=1))  # roc_auc_score(teachers, preds, multi_class='ovo')
