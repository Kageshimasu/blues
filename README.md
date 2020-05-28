# Blues

This python library is useful for creating deep learning models,
supporting Classification, Object Detection, and Semantic Segmentation,
which can be evaluated or inferred using cross-validation.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Examples](#examples)

<a name="features"/>

## Features
* Optimized for high performance
* Easy to apply cross validation
* Easy to conduct augmentations experiment
* Easy to train the following sota models
  - EfficinetNet
  - MobileNet v2 or v3
  - ResNext
  - WideResNet
  - EfficientDet
  - Shelfnet

<a name="installation"/>

## Installation
UNDER CONSTRUCTION...

<a name="examples"/>

## Examples
A standard deep learning situation.
```python
import numpy as np
import shutil
import os
import cv2
import blues
import imgaug.augmenters as iaa


def transpose_for_pytorch(inputs, teachers):
    return inputs.transpose(0, 3, 1, 2) / 255, teachers


# Define Conditions
batch_size = 27
num_classes = 4
epoch = 5
result_path = 'outputs'
width = 128
height = 128
data_size = 81

# Make DummyData
image_root_dir = 'raw'
os.makedirs(image_root_dir)
dummy_inputs = []
dummy_teachers = []
for i in range(data_size):
    rand_image = np.random.rand(width, height, 3)
    image_name = '{}.png'.format(i)
    image_path = os.path.join(image_root_dir, image_name)
    cv2.imwrite(image_path, rand_image)
    dummy_inputs.append(image_path)
    dummy_teachers.append(np.random.randint(0, num_classes))

# Define Callback Functions if you need
transformers = [
    transpose_for_pytorch
]
callback_functions = [
    blues.visualizer.show_image_for_classification
]

# Define Data Augmentations
seq = iaa.Sequential([
    iaa.Crop(),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.Cutout(),
    iaa.Multiply()
])
augmentor = blues.augmentors.ClassificationDataAugmentor(seq)

# Define Models
learning_dir = {
    'fold1': blues.models.classifications.ResNext(num_classes),
    'fold2': blues.models.classifications.WideResNet(num_classes),
    'fold3': blues.models.classifications.MobileNetV2(num_classes),
}
training_table = blues.tables.TrainingTable(learning_dir)

# Define a Dataset
dataset = blues.datasets.ClassificationDataset(
    dummy_inputs,
    dummy_teachers,
    batch_size,
    blues.resizer.ClassificationResizer((width, height)),
    transformers=transformers,
    augmentor=augmentor
)

# RUN!!!
trainer = blues.trainers.XTrainer(
    training_table,
    dataset,
    epoch,
    result_path,
    blues.metrics.accuracy,
    callback_functions=callback_functions,
    evaluate=True
)
trainer.run()
```
