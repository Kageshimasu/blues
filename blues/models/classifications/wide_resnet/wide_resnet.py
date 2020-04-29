import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ....base.base_model import BaseModel


class WideResNet(BaseModel):

    def __init__(self, num_classes, layer=50, pretrained=False, lr=1e-4):
        super().__init__()
        if layer not in [50, 101]:
            raise ValueError('Layer must be 50 or 101')
        self._num_classes = num_classes
        if layer == 50:
            _model = models.wide_resnet50_2
        elif layer == 101:
            _model = models.wide_resnet101_2
        if pretrained:
            self._model = _model(pretrained=True)
        else:
            self._model = _model(pretrained=False, num_classes=num_classes)
        self._layer = layer
        self._optimizer = optim.Adam(self._model.parameters(), lr)
        self._criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._model.cuda()
            self._criterion.cuda()

    def fit(self, inputs, teachers):
        self._model.train()
        output = self._model(inputs)
        loss = self._criterion(output, teachers)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return float(loss)

    def predict(self, inputs):
        self._model.eval()
        with torch.no_grad():
            output = nn.Softmax(dim=1)(self._model(inputs)[:, :self._num_classes])
            pred_ids = output.cpu().numpy()
        return pred_ids

    def save_weight(self, save_path):
        dict_to_save = {
            'num_class': self._num_classes,
            'state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'layer': self._layer
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
        config['model_name'] = 'WideResNet'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer.__class__.__name__
        config['layer'] = self._layer
        return config
