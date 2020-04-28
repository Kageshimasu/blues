import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ....base.base_model import BaseModel
from .mobilenetv3_lib.model import mobilenetv3


class MobileNetV3(BaseModel):

    def __init__(self, num_classes, lr=1e-4, mode='small'):
        super().__init__()
        self._num_classes = num_classes
        self._mode = mode
        self._model = mobilenetv3(num_classes=num_classes, mode=mode)
        # state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        self._optimizer = optim.SGD(self._model.parameters(), lr=1e-4, momentum=0.9)
        self._criterion = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self._model.cuda()
            self._criterion.cuda()

    def fit(self, inputs, teachers):
        self._model.train()
        inputs = torch.Tensor(inputs).cuda().float()
        teachers = torch.Tensor(teachers).cuda().long()

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
            inputs = torch.Tensor(inputs).cuda().float()
            output = self._model(inputs)
            pred_ids = output.cpu().numpy()
        return pred_ids

    def save_weight(self, save_path):
        dict_to_save = {
            'num_class': self._num_classes,
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
        return self

    def get_model_config(self):
        config = {}
        config['model_name'] = 'MobileNetV3'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer
        config['mode'] = self._mode
        return config
