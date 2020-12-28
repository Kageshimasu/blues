import torch
import numpy as np

from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, inputs: torch.Tensor, teachers: torch.Tensor) -> float:
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def save_weight(self, save_path: str) -> object:
        pass

    @abstractmethod
    def get_model_config(self) -> dict:
        pass

    def callback_per_epoch(self):
        pass

