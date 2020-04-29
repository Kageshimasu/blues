import numpy as np
from abc import ABCMeta, abstractmethod


class BaseResizer(metaclass=ABCMeta):

    def __init__(self, out_size: list or tuple):
        """
        :param out_size: (width, height)
        """
        self._out_size = out_size

    @abstractmethod
    def __call__(self, inputs, teachers) -> tuple:
        """
        :param inputs:
        :param teachers:
        :return: (inputs, teachers)
        """
        pass
