from abc import ABCMeta, abstractmethod

from .base_dataset import BaseDataset
from ..tables.predicting_table import PredictingTable
from ..tables.annotation_table import AnnotationTable


class BasePredictor(metaclass=ABCMeta):

    def __init__(
            self, predicting_table: PredictingTable, annotation_table: AnnotationTable, pred_dataset: BaseDataset, result_path: str):
        self._predicting_table = predicting_table
        self._annotation_table = annotation_table
        self._pred_dataset = pred_dataset
        self._result_path = result_path

    @abstractmethod
    def run(self):
        pass
