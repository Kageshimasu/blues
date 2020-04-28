from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, inputs, teachers):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass

    @abstractmethod
    def save_weight(self, save_path: str):
        pass

    @abstractmethod
    def get_model_config(self):
        pass
