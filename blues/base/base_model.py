from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, train_data, test_data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def save_weight(self, save_path):
        pass

    @abstractmethod
    def get_model_config(self):
        pass
