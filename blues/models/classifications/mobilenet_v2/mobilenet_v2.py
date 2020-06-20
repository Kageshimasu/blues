import torch.optim as optim
import torch
import torchvision.models as models

from ....base.base_model import BaseModel


class MobileNetV2(BaseModel):

    def __init__(self, num_classes, pretrained=True, lr=1e-4):
        super().__init__()
        self._num_classes = num_classes
        if pretrained:
            self._model = models.mobilenet_v2(pretrained=True)
        else:
            self._model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
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
            output = self._model(inputs)[:, :self._num_classes]
            pred_ids = output.cpu()
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
        config['model_name'] = 'MobileNetV2'
        config['num_classes'] = self._num_classes
        config['optimizer'] = self._optimizer.__class__.__name__
        return config
