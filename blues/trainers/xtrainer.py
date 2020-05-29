from tqdm import tqdm
from collections import OrderedDict
import os
import json
from typing import List, Dict
import matplotlib.pyplot as plt

from blues.tables.training_table import TrainingTable
from ..base.base_trainer import BaseTrainer
from ..base.base_dataset import BaseDataset
from ..base.base_kfolder import BaseKFolder
from ..kfold.simple_kfold import KFolder
from ..plotter.metric_store import MetricStore
from ..plotter.curve_plotter import CurvePlotter
from ..plotter.box_plotter import BoxPlotter


class XTrainer(BaseTrainer):
    _FOLD_NAME = 'fold_{}'
    _CALL_FUNC_PER = 10

    def __init__(
            self, learning_table: TrainingTable, train_dataset: BaseDataset, num_epochs: int, result_path: str,
            metrics: List[callable], test_dataset: BaseDataset = None, callback_functions: List[callable] = None,
            kfolder: BaseKFolder.__class__ = KFolder, evaluate=True):
        """
        XTrainer is a module to train models by cross validation.
        :param learning_table:
        :param train_dataset:
        :param num_epochs:
        :param result_path:
        :param metrics:
        :param test_dataset:
        :param callback_functions:
        :param kfolder:
        :param evaluate:
        """
        super().__init__(learning_table, train_dataset, result_path, num_epochs, test_dataset, callback_functions)
        self._kf = kfolder
        self._metrics = metrics
        self._metric_list = []
        self._evaluate = evaluate
        self._metric_store = MetricStore([metric_func.__name__ for metric_func in self._metrics], self._result_path)

    def run(self):
        """
        start learning
        :return:
        """
        models = []
        config = {}
        os.makedirs(self._result_path, exist_ok=False)

        for fold_num, model in self._learning_table:
            print('FOLD: {}, MODEL: {} LOADED'.format(fold_num, model))
            config[fold_num] = model.get_model_config()
            models.append(model)

        print(config)
        with open(os.path.join(self._result_path, 'config.json'), 'w') as f:
            json.dump(config, f)

        kf = self._kf(self._train_dataset, n_splits=len(self._learning_table))
        for k, (train_dataset, valid_dataset) in enumerate(kf):
            fold_path = os.path.join(self._result_path, self._FOLD_NAME.format(k))
            os.makedirs(fold_path, exist_ok=False)
            model_name = models[k].__class__.__name__
            print('\n### FOLD: {}, MODEL: {}'.format(k + 1, model_name))

            for epoch in range(self._num_epochs):
                with tqdm(total=len(train_dataset) // train_dataset.get_batch_size()) as iter_bar:
                    losses = []
                    train_metrics_dict = {}

                    for iter, data in enumerate(train_dataset):
                        inputs_numpy, teachers_numpy = data.get_inputs_on_numpy(), data.get_teachers_on_numpy()
                        inputs_tensor, teachers_tensor = data.get_inputs_on_torch(), data.get_teachers_on_torch()
                        loss = models[k].fit(inputs_tensor, teachers_tensor)
                        preds = models[k].predict(inputs_tensor)
                        current_metric_dict = {}
                        for metric_func in self._metrics:
                            metric_name = metric_func.__name__
                            metric_val = metric_func(teachers_numpy, preds)
                            if metric_name not in train_metrics_dict:
                                train_metrics_dict[metric_name] = []
                            train_metrics_dict[metric_name].append(metric_val)
                            current_metric_dict[metric_name] = metric_val

                        losses.append(loss)
                        if iter % self._CALL_FUNC_PER == 0:
                            if self._callback_functions is not None:
                                for callback in self._callback_functions:
                                    callback(inputs_numpy, preds, teachers_numpy)

                        iter_bar.set_description('ITERATOR')
                        iter_bar.set_postfix(OrderedDict(
                            FOLD='{}/{}'.format(k, len(kf) - 1),
                            EPOCH='{}/{}'.format(epoch, self._num_epochs - 1),
                            LOSS=loss,
                            METRIC=current_metric_dict
                        ))
                        iter_bar.update(1)
                eval_metrics_dict = {}
                valid_metrics_mean_dict = {'none': 0}

                if self._evaluate:
                    for data in valid_dataset:
                        inputs_numpy, teachers_numpy = data.get_inputs_on_numpy(), data.get_teachers_on_numpy()
                        inputs_torch = data.get_inputs_on_torch()
                        preds = models[k].predict(inputs_torch)
                        for metric_func in self._metrics:
                            metric_name = metric_func.__name__
                            if metric_name not in eval_metrics_dict:
                                eval_metrics_dict[metric_name] = []
                            eval_metrics_dict[metric_name].append(metric_func(teachers_numpy, preds))

                    valid_metrics_mean_dict = self._calc_metric_mean(eval_metrics_dict)
                    train_metrics_mean_dict = self._calc_metric_mean(train_metrics_dict)
                    self._metric_list.append(valid_metrics_mean_dict)
                    print('TRAIN MEAN LOSS: {}'.format(sum(losses) / len(losses)))
                    print('TRAIN MEAN METRIC: {}'.format(train_metrics_mean_dict))
                    print('VALID MEAN METRIC: {}\n'.format(valid_metrics_mean_dict))
                    self._metric_store.append(model_name, k, epoch, train_metrics_mean_dict, valid_metrics_mean_dict)

                models[k].save_weight(
                    os.path.join(fold_path, 'epoch_{}_metric_{}.pth'.format(
                        epoch,
                        [metric_name + '_' + str(valid_metrics_mean_dict[metric_name]) + '_'
                         for metric_name in valid_metrics_mean_dict.keys()]
                    )))

        if self._evaluate:
            curve_plotter = CurvePlotter(self._metric_store)
            box_plotter = BoxPlotter(self._metric_store)
            curve_plotter.plot(self._result_path)
            box_plotter.plot(self._result_path)
            plt.show()

    @staticmethod
    def _calc_metric_mean(metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:
        """
        :param metrics_dict:
        :return:
        """
        ret_dict = {}
        for metric_name in metrics_dict.keys():
            metric_list = metrics_dict[metric_name]
            ret_dict[metric_name] = float(sum(metric_list) / len(metric_list))
        return ret_dict
