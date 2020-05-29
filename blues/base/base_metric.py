from abc import ABCMeta, abstractmethod
import numpy as np


class BaseMetric(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def compute_value_for_one_batch(self, teacher: np.ndarray, pred: np.ndarray) -> any:
        pass

    @abstractmethod
    def append_metric(self):
        pass

    @abstractmethod
    def compute_mean(self) -> any:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def clear(self):
        pass
