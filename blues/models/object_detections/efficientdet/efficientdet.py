import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

from ....base.base_model import BaseModel
from .efficientdet_lib.models.efficientdet import EfficientDet
from .efficientdet_lib.utils import EFFICIENTDET
from ..._model_consts import _ModelConst
from .efficientdet_lib.models.module import bias_init_with_prob, normal_init


class EfficientDetector(BaseModel):

    def __init__(self, num_classes, network='efficientdet-d0', lr=1e-4, score_threshold=0.2, max_detections=100):
        super().__init__()
        self._model = self._initialize_model(num_classes, network)
        self._num_classes = num_classes
        self._network = network
        self._optimizer = optim.AdamW(self._model.parameters(), lr=lr)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, patience=3, verbose=True)
        self._score_threshold = score_threshold
        self._max_detections = max_detections

    @staticmethod
    def _initialize_model(num_classes, network):
        return EfficientDet(num_classes=num_classes,
                                   network=network,
                                   W_bifpn=EFFICIENTDET[network]['W_bifpn'],
                                   D_bifpn=EFFICIENTDET[network]['D_bifpn'],
                                   D_class=EFFICIENTDET[network]['D_class']).cuda()

    def fit(self, inputs, teachers):
        self._model.train()
        self._model.is_training = True
        self._model.freeze_bn()
        self._optimizer.zero_grad()

        classification_loss, regression_loss = self._model([inputs, teachers.float()])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            return 0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.1)
        self._optimizer.step()
        self._optimizer.zero_grad()
        return float(loss)

    def predict(self, inputs):
        self._model.eval()
        self._model.is_training = False
        batch_size = inputs.shape[0]
        all_detections = [[None for _ in range(self._num_classes)] for _ in range(batch_size)]

        with torch.no_grad():
            for i in range(batch_size):
                image = torch.Tensor(inputs[i]).cuda().float().unsqueeze(0)
                scores, labels, boxes = self._model(image)
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()
                boxes = boxes.cpu().numpy()
                indices = np.where(scores > self._score_threshold)[0]
                if indices.shape[0] > 0:
                    scores = scores[indices]
                    scores_sort = np.argsort(-scores)[:self._max_detections]
                    # select detections
                    image_boxes = boxes[indices[scores_sort], :]
                    image_scores = scores[scores_sort]
                    image_labels = labels[indices[scores_sort]]
                    image_detections = np.concatenate([
                        image_boxes,
                        np.expand_dims(image_scores, axis=1),
                        np.expand_dims(image_labels, axis=1)
                    ], axis=1)

                    # copy detections to all_detections
                    for label in range(self._num_classes):
                        all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

        return all_detections

    def save_weight(self, save_path):
        dict_to_save = {
            _ModelConst.NUM_CLASSES: self._num_classes,
            _ModelConst.NETWORK: self._network,
            _ModelConst.STATE_DICT: self._model.state_dict(),
            _ModelConst.OPTIMIZER: self._optimizer.state_dict(),
        }
        torch.save(dict_to_save, save_path)

    def load_weight(self, weight_path, fine_tune=True):
        params = torch.load(weight_path)
        print('The pretrained weight is loaded')
        print('Network: {}'.format(params[_ModelConst.NETWORK]))
        self._network = params[_ModelConst.NETWORK]
        pretrained_num_classes = params[_ModelConst.NUM_CLASSES]
        self._model = self._initialize_model(pretrained_num_classes, self._network)
        self._model.load_state_dict(params[_ModelConst.STATE_DICT])
        self._optimizer.load_state_dict(params[_ModelConst.OPTIMIZER])

        if fine_tune:
            self._model.bbox_head.num_classes = self._num_classes
            self._model.bbox_head.cls_out_channels = self._num_classes
            self._model.bbox_head.retina_cls = nn.Conv2d(
                self._model.bbox_head.feat_channels,
                self._model.bbox_head.num_anchors * self._num_classes,
                3,
                padding=1)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(self._model.bbox_head.retina_cls, std=0.01, bias=bias_cls)
        else:
            self._num_classes = pretrained_num_classes
        return self

    def get_model_config(self):
        config = {
            _ModelConst.MODEL_NAME: 'EfficientDetector',
            _ModelConst.NUM_CLASSES: self._num_classes,
            _ModelConst.NETWORK: self._network,
            _ModelConst.STATE_DICT: self._model.state_dict().__class__.__name__
        }
        return config
