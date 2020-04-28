from blues.models.classifications.resnext.resnext import ResNext
from blues.models.classifications.wide_resnet.wide_resnet import WideResNet
from blues.models.classifications.mobilenet_v2.mobilenet_v2 import MobileNetV2
from blues.datasets.classification_dataset import ClassificationDataset
from blues.trainers.xtrainer import XTrainer
from blues.visualizer.classification_visualizer import show_image
from blues.metrics.classification_metrics import accuracy
from blues.resizer.classification_resizer import ClassificationResizer
from blues.tables.learning_table import LearningTable

from compe.plant.dataset_maker import DatasetMaker


def transpose_for_pytorch(inputs, teachers):
    return inputs.transpose(0, 3, 1, 2) / 255, teachers


def main():
    batch_size = 4
    epoch = 3
    num_classes = 4
    width = 512
    height = 512
    result_path = './result'
    dataset_maker = DatasetMaker('raw/images', 'raw/train.csv', 'raw/test.csv')

    # FUNCTIONS
    transformers = [
        ClassificationResizer((width, height)),
        transpose_for_pytorch
    ]
    callback_functions = [
        show_image
    ]

    # MODELS
    learning_dir = {
        'fold1': MobileNetV2(num_classes, pretrained=True),
        'fold2': ResNext(num_classes, pretrained=True, layer=50),
        'fold3': ResNext(num_classes, pretrained=True, layer=101),
        'fold4': WideResNet(num_classes, pretrained=True, layer=50),
        'fold5': WideResNet(num_classes, pretrained=True, layer=101)
    }
    learning_table = LearningTable(learning_dir)

    # DATASET
    dataset = ClassificationDataset(
        dataset_maker.train_inputs, dataset_maker.train_teachers, batch_size, transformers=transformers)

    # RUN!
    trainer = XTrainer(
        learning_table, dataset, epoch, result_path, accuracy, callback_functions=callback_functions, evaluate=True)
    trainer.run()


if __name__ == "__main__":
    main()
