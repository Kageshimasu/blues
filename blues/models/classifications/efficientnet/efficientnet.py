import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

from ....base.base_model import BaseModel
# from .efficientnet_lib.model import EfficientNetPredictor
# from efficientnet_pytorch import EfficientNet
from .efficientnet_lib.lib import Model


class EfficientNet(BaseModel):

    def __init__(self, num_classes, network='efficientnet-b0', lr=0.1, momentum=0.9, weight_decay=1e-4):
        super().__init__()
        self._num_classes = num_classes
        self._model = Model.from_pretrained(network)
        self._network = network
        self._optimizer = optim.SGD(
            self._model.parameters(), lr,
            momentum=momentum,
            weight_decay=weight_decay)
        self._criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._model.cuda()
            self._criterion.cuda()

    def fit(self, inputs, teachers):
        self._model.train()
        # compute output
        output = self._model(inputs)
        loss = self._criterion(output, teachers)
        # compute gradient and do SGD step
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss)

    def predict(self, inputs):
        self._model.eval()
        with torch.no_grad():
            output = self._model(inputs)[:, :self._num_classes]
            pred_ids = output.cpu()
        return pred_ids

    def save_weight(self, save_path):
        dict_to_save = {
            'num_class': self._num_classes,
            'network': self._network,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        params = torch.load(weight_path)
        print('The pretrained weight is loaded')
        print('Num classes: {}'.format(params['num_class']))
        self._num_classes = params['num_class']
        self._model.load_state_dict(params['state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])
        self._network = params['network']
        return self

    def get_model_config(self):
        config = {}
        config['model_name'] = 'EfficientNet'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer.__class__.__name__
        config['network'] = self._network
        return config
