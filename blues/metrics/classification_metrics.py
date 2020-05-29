import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def accuracy(teachers, preds):
    return accuracy_score(teachers, np.argmax(preds, axis=1))


def accuracy_2(teachers, preds):
    return accuracy_score(teachers, np.argmax(preds, axis=1))
