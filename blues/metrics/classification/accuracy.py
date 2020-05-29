from ...base.base_metric import BaseMetric


class Accuracy(BaseMetric):

    def __init__(self):
        pass

    def compute_value_for_one_batch(self, teacher, pred):
        pass

    def get_mean(self) -> any:
        pass

    def get_name(self) -> str:
        pass

    def clear(self):
        pass
