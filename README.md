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
