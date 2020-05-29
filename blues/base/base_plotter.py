from abc import ABCMeta, abstractmethod
from ..plotter.metric_store import MetricStore


class BasePlotter(metaclass=ABCMeta):

    def __init__(self, metric_store: MetricStore):
        self._metric_store = metric_store

    @abstractmethod
    def plot(self, save_path: str):
        pass
