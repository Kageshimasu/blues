import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ....base.base_model import BaseModel


class Flatten(nn.Module):
    def forward(self, x):
        # print(x.view(x.size(0), -1).shape)
        return x.view(x.size(0), -1)


class SimpleCNNBlock(nn.Module):

    def __init__(self, in_c, out_c, k_size):
        super(SimpleCNNBlock, self).__init__()
        self._conv = nn.Conv2d(in_c, out_c, k_size)
        self._relu = nn.ReLU()
        # self._bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self._relu(self._conv(x))


class SimpleDenseBlock(nn.Module):

    def __init__(self, in_c, out_c, activation=nn.ReLU):
        super(SimpleDenseBlock, self).__init__()
        # self._bn = nn.BatchNorm1d(in_c)
        self._fn = nn.Linear(in_c, out_c)
        self._relu = activation()

    def forward(self, x):
        return self._relu(self._fn(x))


model = torch.nn.Sequential(
            SimpleCNNBlock(3, 32, 3),
            SimpleCNNBlock(32, 32, 3),
            SimpleCNNBlock(32, 32, 5),
            # nn.Dropout2d(0.4),
            SimpleCNNBlock(32, 64, 3),
            SimpleCNNBlock(64, 64, 3),
            SimpleCNNBlock(64, 64, 5),
            Flatten(),
            SimpleDenseBlock(147456, 1024),
            SimpleDenseBlock(1024, 512),
            SimpleDenseBlock(512, 10, activation=nn.Softmax)
        )


class SimpleNet(BaseModel):

    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self._num_classes = num_classes
        self._model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        self._optimizer = optim.Adam(self._model.parameters(),
                                     lr)  # optim.SGD(self._model.parameters(), lr=0.1, momentum=0.9)
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
        config['model_name'] = 'SimpleNet'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer.__class__.__name__
        return config
