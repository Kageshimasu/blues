import pytest
import numpy as np
import shutil
import os
import cv2
import blues


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
    ##### 学習条件定義 #####
    batch_size = 27
    num_classes = 4
    epoch = 5
    result_path = 'outputs'
    width = 128
    height = 128
    data_size = 81

    ##### ダミーデータ作成 #####
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

    ##### コールバック定義 #####
    transformers = [
        blues.resizer.ClassificationResizer((width, height)),
        transpose_for_pytorch
    ]
    callback_functions = [
        blues.visualizer.show_image_for_classification
    ]

    ##### モデル定義 #####
    learning_dir = {
        'fold1': model(num_classes),
        'fold2': model(num_classes),
        'fold3': model(num_classes),
    }
    learning_table = blues.tables.TrainingTable(learning_dir)

    ##### データセット定義 #####
    dataset = blues.datasets.ClassificationDataset(
        dummy_inputs, dummy_teachers, batch_size, transformers=transformers)

    ##### RUN! #####
    trainer = blues.trainers.XTrainer(
        learning_table,
        dataset,
        epoch,
        result_path,
        blues.metrics.accuracy,
        callback_functions=callback_functions,
        evaluate=True
    )
    trainer.run()

    ##### 後始末 #####
    shutil.rmtree(image_root_dir)
    shutil.rmtree(result_path)
