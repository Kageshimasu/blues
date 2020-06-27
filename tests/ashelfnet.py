import blues
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

COLOR_DICT = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128]
}


class PilToLabel:

    def __init__(self):
        pass

    def __call__(self, img):
        targets = np.array(img.convert("RGB"), dtype=np.uint8)
        return self._voc_label_indices(targets)

    def _voc_label_indices(self, colormap):
        colormap2label = self._build_colormap2label()
        colormap = colormap.astype(np.int32)
        idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
               + colormap[:, :, 2])
        return np.array(colormap2label[idx], dtype=np.uint8)

    def _build_colormap2label(self):
        """Build an RGB color to label mapping for segmentation."""
        colormap2label = np.zeros(256 ** 3)
        for i, colormap in enumerate(VOC_COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label


def labels_to_imgs(labels):
    labels = np.argmax(np.array(labels), axis=1)
    img = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2], 3))
    for key in COLOR_DICT.keys():
        img[labels == key] = COLOR_DICT[key]
    return img


def main():
    # y = torch.randint(0, 20, (16, 256, 256))
    # labels_to_imgs(y)
    # exit()

    model = blues.models.semantic_segmentations.ShelfNet
    epoch = 100
    num_classes = 22
    batch_size = 24
    crop_size = [224, 224]
    color_dict = {}

    shelfnet = model(num_classes, batch_size, crop_size)

    # visualizer = blues.visualizer.SemanticSegmentationVisualizer(color_dict, )
    transform = transforms.Compose([
        transforms.Resize((crop_size[0], crop_size[1])),
        transforms.ToTensor()
    ])
    transform_target = transforms.Compose([
        transforms.Resize((crop_size[0], crop_size[1])),
        PilToLabel(),
        transforms.ToTensor()
    ])
    voc_data = torchvision.datasets.VOCSegmentation(
        './tests/VOCdevkit/', download=True, transform=transform, target_transform=transform_target)
    dataloader = torch.utils.data.DataLoader(
        voc_data, batch_size=batch_size, shuffle=True)

    for e in range(epoch):
        for data in dataloader:
            inputs, targets = data
            if inputs.shape[0] != 16:
                continue
            # to cuda
            inputs = inputs.cuda()
            targets = targets.cuda().long().squeeze(1)

            # fit and pred
            loss = shelfnet.fit(inputs, targets)
            preds_np = shelfnet.predict(inputs)

            # visualize
            # inputs_np = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            # targets_np = labels_to_imgs(targets.cpu().numpy())
            preds_img = labels_to_imgs(preds_np)
            # cv2.imshow('input', inputs_np[0])
            # cv2.imshow('target', targets_np[0])
            print('a')
            cv2.imshow('pred', preds_img[0])
            cv2.waitKey(1)
            print(loss)


if __name__ == '__main__':
    main()
