import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Dict, List


class _MetricStoreConst:
    MODEL_NAME = 'model_name'
    FOLD = 'fold'
    EPOCH = 'epoch'
    TRAIN_OR_VALID = 'train_or_valid'
    TRAIN = 'train'
    VALID = 'valid'
    CSV_NAME_TO_SAVE = 'metrics.csv'
    FIG_NAME_TO_SAVE = ''

    class ConstError(TypeError):
        pass

    def get_all_consts(self) -> List[str]:
        return [self.MODEL_NAME, self.FOLD, self.EPOCH, self.TRAIN_OR_VALID]

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value


class MetricStore:

    def __init__(self, metric_names: List[str], output_path: str):
        """
        :param metric_names: ["accuracy", "mean_squared_error", ...]
        """
        metric_dict = {}
        self.consts = _MetricStoreConst()
        for metric_name in metric_names:
            if metric_name in self.consts.get_all_consts():
                raise ValueError('Reserved words are included, got {}'.format(metric_name))
            metric_dict[metric_name] = []
        self._df_dict = {}
        for const in self.consts.get_all_consts():
            self._df_dict[const] = []
        self._df_dict.update(metric_dict)
        self._metric_names = metric_names
        self._output_path = output_path

    def append(
            self, model_name: str, fold: int, epoch: int, train_metrics: Dict[str, float], valid_metrics: Dict[str, float]):
        """
        :param model_name:
        :param fold:
        :param epoch:
        :param train_metrics: Dict[metric_name, metric_mean_value]
        :param valid_metrics: Dict[metric_name, metric_mean_value]
        :return:
        """
        if set(train_metrics.keys()) != set(valid_metrics.keys()):
            raise ValueError('Train metrics keys must be equal to valid ones')
        elif set(train_metrics.keys()) != set(self._metric_names) or set(valid_metrics.keys()) != set(self._metric_names):
            raise ValueError('Metrics keys must be equal to the metric names, which is put at the constructor')

        for train_or_valid in [self.consts.TRAIN, self.consts.VALID]:
            self._df_dict[self.consts.EPOCH].append(epoch)
            self._df_dict[self.consts.MODEL_NAME].append(model_name)
            self._df_dict[self.consts.FOLD].append(str(fold))
            self._df_dict[self.consts.TRAIN_OR_VALID].append(train_or_valid)
            for metric_name in self._metric_names:
                if train_or_valid == self.consts.TRAIN:
                    self._df_dict[metric_name].append(train_metrics[metric_name])
                elif train_or_valid == self.consts.VALID:
                    self._df_dict[metric_name].append(valid_metrics[metric_name])

    def save_df_as_csv(self):
        self.get_dict_as_df().to_csv(os.path.join(self._output_path, self.consts.CSV_NAME_TO_SAVE), index=False)

    def load_df_as_csv(self, output_path: str):
        df = pd.read_csv(os.path.join(output_path, self.consts.CSV_NAME_TO_SAVE))
        self._df_dict = df.to_dict()

    def get_dict_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._df_dict)

    def get_metric_names(self) -> List[str]:
        return self._metric_names

    def get_num_metrics(self) -> int:
        return len(self._metric_names)
