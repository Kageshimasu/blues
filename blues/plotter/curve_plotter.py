import matplotlib.pyplot as plt
import seaborn as sns
import os

from .metric_store import MetricStore
from ..base.base_plotter import BasePlotter


class CurvePlotter(BasePlotter):
    _SAVE_NAME = 'curve_plot.png'

    def __init__(self, metric_store: MetricStore):
        super().__init__(metric_store)
        self._fig, self._axes = plt.subplots(1, metric_store.get_num_metrics())

    def plot(self, save_path: str):
        df_grouped_by_fold = self._metric_store.get_dict_as_df().groupby(
            [self._metric_store.consts.EPOCH, self._metric_store.consts.TRAIN_OR_VALID], as_index=False).mean()

        metric_names = self._metric_store.get_metric_names()
        for i, ax in enumerate(self._axes):
            sns.lineplot(
                ax=ax,
                data=df_grouped_by_fold,
                x=self._metric_store.consts.EPOCH,
                y=metric_names[i],
                hue=self._metric_store.consts.TRAIN_OR_VALID
            )

        self._fig.savefig(os.path.join(save_path, self._SAVE_NAME))
