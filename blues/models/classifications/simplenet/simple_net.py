import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ....base.base_model import BaseModel

# self.conv1 = nn.Conv2d(3, 32, 5)
# self.conv2 = nn.Conv2d(32, 32, 5)
# self.pool = nn.MaxPool2d(2, 2)
# self.conv3 = nn.Conv2d(32, 64, 5)
# self.conv4 = nn.Conv2d(64, 64, 5)
# self.fc2 = nn.Linear(256, 256)
# self.fc3 = nn.Linear(256, num_classes)


class NetModule(nn.Module):
    def __init__(self, num_classes):
        super(NetModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(86528, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(32, -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleNet(BaseModel):

    def __init__(self, num_classes, lr=1e-3, momentum=0.9):
        super().__init__()
        self._num_classes = num_classes
        self._model = NetModule(num_classes)
        self._optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=momentum)
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
