from tqdm import tqdm
from collections import OrderedDict
import os
import json

from blues.tables.learning_table import TrainingTable
from ..base.base_trainer import BaseTrainer
from ..base.base_dataset import BaseDataset
from ..base.base_kfolder import BaseKFolder
from ..kfold.simple_kfold import KFolder


class XTrainer(BaseTrainer):
    _FOLD_NAME = 'fold_{}'
    _CALL_FUNC_PER = 10

    def __init__(
            self, learning_table: TrainingTable, train_dataset: BaseDataset, num_epochs: int, result_path: str,
            metric_function: callable, test_dataset: BaseDataset = None, callback_functions: list = None,
            kfolder: BaseKFolder.__class__ = KFolder, evaluate=True):
        """
        XTrainer is a module to train models by cross validation.
        :param learning_table:
        :param train_dataset:
        :param num_epochs:
        :param result_path:
        :param metric_function:
        :param test_dataset:
        :param callback_functions:
        :param kfolder:
        :param evaluate:
        """
        super().__init__(learning_table, train_dataset, result_path, num_epochs, test_dataset, callback_functions)
        self._kf = kfolder
        self._metric_function = metric_function
        self._metric_list = []
        self._evaluate = evaluate

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
            print('\n### FOLD: {}, MODEL: {}'.format(k + 1, models[k].__class__.__name__))

            for epoch in range(self._num_epochs):
                with tqdm(total=len(train_dataset) // train_dataset.get_batch_size()) as iter_bar:
                    losses = []
                    train_metrics = []

                    for iter, data in enumerate(train_dataset):
                        inputs_numpy, teachers_numpy = data.get_inputs_on_numpy(), data.get_teachers_on_numpy()
                        inputs_tensor, teachers_tensor = data.get_inputs_on_torch(), data.get_teachers_on_torch()
                        loss = models[k].fit(inputs_tensor, teachers_tensor)
                        preds = models[k].predict(inputs_tensor)
                        metric = self._metric_function(teachers_numpy, preds)
                        losses.append(loss)
                        train_metrics.append(metric)
                        if iter % self._CALL_FUNC_PER == 0:
                            if self._callback_functions is not None:
                                for callback in self._callback_functions:
                                    callback(inputs_numpy, preds, teachers_numpy)

                        iter_bar.set_description('ITERATOR')
                        iter_bar.set_postfix(OrderedDict(
                            FOLD='{}/{}'.format(k, len(kf) - 1),
                            EPOCH='{}/{}'.format(epoch, self._num_epochs - 1),
                            LOSS=loss,
                            METRIC=metric
                        ))
                        iter_bar.update(1)
                eval_metrics = []
                valid_mean_metric = 0

                if self._evaluate:
                    for data in valid_dataset:
                        inputs_numpy, teachers_numpy = data.get_inputs_on_numpy(), data.get_teachers_on_numpy()
                        inputs_torch = data.get_inputs_on_torch()
                        preds = models[k].predict(inputs_torch)
                        metric = self._metric_function(teachers_numpy, preds)
                        eval_metrics.append(metric)
                    valid_mean_metric = sum(eval_metrics) / len(eval_metrics)
                    self._metric_list.append(valid_mean_metric)
                    print('TRAIN MEAN LOSS: {}'.format(sum(losses) / len(losses)))
                    print('TRAIN MEAN METRIC: {}'.format(sum(train_metrics) / len(train_metrics)))
                    print('VALID MEAN METRIC: {}\n'.format(valid_mean_metric))

                models[k].save_weight(
                    os.path.join(fold_path, 'epoch_{}_metric_{}.pth'.format(
                        epoch, valid_mean_metric)))

                # TODO: test datasetの推論とグラフ描画までやったらいいね
