import torch.optim as optim
import torch

from resnest.torch import resnest50, resnest101, resnest200
from ....base.base_model import BaseModel


class ResNest(BaseModel):

    def __init__(self, num_classes, layer=50, pretrained=True, channel=1, is_sigmoid=True):
        super().__init__()
        if layer not in [50, 101, 200]:
            raise ValueError('Layer must be 50 or 101 or 200')
        if channel not in [1, 3]:
            raise ValueError('Channel must be 1 or 3')
        self._num_classes = num_classes
        if layer == 50:
            _model = resnest50
        elif layer == 101:
            _model = resnest101
        elif layer == 200:
            _model = resnest200

        self._model = _model(pretrained=pretrained)
        self._model.fc = torch.nn.Sequential(torch.nn.Linear(2048, num_classes))
        self._layer = layer
        self._optimizer = optim.SGD(self._model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=7, gamma=0.4)
        self._criterion = torch.nn.CrossEntropyLoss()

        if channel == 1:
            self._model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if is_sigmoid:
            self._model.fc.add_module('sigmoid', torch.nn.Sigmoid())
            pos_weights = torch.ones(num_classes)
            pos_weights = pos_weights * num_classes
            self._criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        if torch.cuda.is_available():
            self._model.cuda()
            self._criterion.cuda()

    def fit(self, inputs, teachers):
        self._model.train()
        output = self._model(inputs)
        loss = self._criterion(output, teachers.float())
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
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'layer': self._layer,
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path):
        params = torch.load(weight_path)
        print('The pretrained weight is loaded')
        print('Num classes: {}'.format(params['num_class']))
        self._num_classes = params['num_class']
        self._model.load_state_dict(params['state_dict'])
        self._optimizer.load_state_dict(params['optimizer'])
        # self._layer = params['layer']
        return self

    def get_model_config(self):
        config = {}
        config['model_name'] = 'ResNext'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer.__class__.__name__
        config['layer'] = self._layer
        return config

    def callback_per_epoch(self):
        self._scheduler.step()
