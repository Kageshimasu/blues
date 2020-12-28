import matplotlib.pyplot as plt
import seaborn as sns
import os

from .metric_store import MetricStore
from ..base.base_plotter import BasePlotter


class BoxPlotter(BasePlotter):
    _SAVE_NAME = 'box_plot.png'

    def __init__(
            self, metric_store: MetricStore, eval_from: int = 0, eval_by: int = None, grouped_by: str = 'model_name'):
        """
        :param metric_store:
        :param eval_from:
        :param eval_by:
        :param grouped_by:
        """
        super().__init__(metric_store)
        if grouped_by not in self._metric_store.consts.get_all_consts():
            raise ValueError('The column name for grouping is not included in the metric store columns')
        if eval_from < 0:
            raise ValueError('should be evaluated from 0 epoch but got under 0')
        if eval_by is not None and eval_by <= eval_from:
            raise ValueError('Eval by must outnumber Eval from')
        self._eval_from = eval_from
        if eval_by is None:
            self._eval_by = self._metric_store.get_dict_as_df()[self._metric_store.consts.EPOCH].max()
        else:
            self._eval_by = eval_by
        # TODO: モデルごとboxplotの結果を見るのはあり？
        self._grouped_by = grouped_by
        self._fig, self._axes = plt.subplots(1, metric_store.get_num_metrics())
        if metric_store.get_num_metrics() == 1:
            self._axes = [self._axes]

    def plot(self, save_path: str):
        df_grouped = self._metric_store.get_dict_as_df().groupby(
            [self._metric_store.consts.EPOCH, self._metric_store.consts.TRAIN_OR_VALID],
            as_index=False).mean()
        df_extracted_for_valid = df_grouped[
            (df_grouped[self._metric_store.consts.TRAIN_OR_VALID] == self._metric_store.consts.VALID)
            & (df_grouped[self._metric_store.consts.EPOCH] >= self._eval_from)
            & (df_grouped[self._metric_store.consts.EPOCH] <= self._eval_by)
        ]

        metric_names = self._metric_store.get_metric_names()
        for i, ax in enumerate(self._axes):
            sns.boxplot(
                ax=ax,
                data=df_extracted_for_valid,
                y=metric_names[i],
                hue=self._metric_store.consts.TRAIN_OR_VALID
            )

        self._fig.savefig(os.path.join(save_path, self._SAVE_NAME))
