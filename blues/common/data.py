import torch
import numpy as np


class Data:

    def __init__(self, inputs: np.ndarray, teachers: torch.Tensor, file_name: list, scale: float = 1, is_cuda=True):
        """
        :param inputs:
        :param teachers:
        :param file_name:
        :param scale:
        """
        self._inputs = inputs
        self._teachers = teachers
        self._file_name = file_name
        self._scale = scale
        self._is_cuda = is_cuda

    def get_inputs_on_numpy(self) -> np.ndarray:
        return self._inputs

    def get_teachers_on_numpy(self) -> np.ndarray:
        return self._teachers

    def get_inputs_on_torch(self) -> torch.Tensor:
        if self._is_cuda:
            return torch.Tensor(self._inputs).cuda().float()
        return torch.Tensor(self._inputs).float()

    def get_teachers_on_torch(self) -> torch.Tensor:
        if self._is_cuda:
            return torch.Tensor(self._teachers).cuda().long()
        return torch.Tensor(self._teachers).long()

    def get_file_names(self) -> list:
        return self._file_name

    def get_scale(self) -> float:
        return self._scale
