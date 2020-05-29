import blues
import torch
import torch.nn as nn
import pytest


@pytest.mark.parametrize(
    "model", [
        blues.models.semantic_segmentations.ShelfNet
    ]
)
def test_segmentation(model):
    num_classes = 12
    batch_size = 16
    crop_size = [224, 224]

    net = model(num_classes, batch_size, crop_size)
    x = torch.randn(batch_size, 3, crop_size[0], crop_size[1])
    y = torch.randn(batch_size, num_classes, crop_size[0], crop_size[1])
    print(net.fit(x, y))
    print(net.predict(x).shape)

