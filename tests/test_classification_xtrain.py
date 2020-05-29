import pytest
import numpy as np
import shutil
import os
import cv2
import blues
import imgaug.augmenters as iaa


def transpose_for_pytorch(inputs, teachers):
    return inputs.transpose(0, 3, 1, 2) / 255, teachers


@pytest.mark.parametrize(
    "model", [
        blues.models.classifications.ResNext,
        blues.models.classifications.WideResNet,
        blues.models.classifications.MobileNetV2,
    ]
)
def test_classification(model):
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
        # iaa.Fliplr(),
    ])
    augmentor = blues.augmentors.ClassificationDataAugmentor(seq)

    # Define Models
    learning_dir = {
        'fold1': model(num_classes),
        'fold2': model(num_classes),
        'fold3': model(num_classes),
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
        [blues.metrics.accuracy],
        callback_functions=callback_functions,
        evaluate=True
    )
    trainer.run()

    shutil.rmtree(image_root_dir)
    shutil.rmtree(result_path)
