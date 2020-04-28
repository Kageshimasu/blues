from blues.models.classifications.resnext.resnext import ResNext
from blues.models.classifications.wide_resnet.wide_resnet import WideResNet
from blues.models.classifications.mobilenet_v2.mobilenet_v2 import MobileNetV2
from blues.datasets.classification_dataset import ClassificationDataset
from blues.visualizer.classification_visualizer import show_image
from blues.resizer.classification_resizer import ClassificationResizer
from blues.predictors.classification_predictor import ClassificationPredictor
from blues.tables.predicting_table import PredictingTable
from blues.tables.annotation_table import AnnotationTable

from compe.plant.dataset_maker import DatasetMaker


def transpose_for_pytorch(inputs, teachers):
    return inputs.transpose(0, 3, 1, 2) / 255, teachers


def main():
    batch_size = 4
    num_classes = 4
    width = 512
    height = 512
    result_path = './submission'

    dataset_maker = DatasetMaker('raw/images', 'raw/train.csv', 'raw/test.csv')

    # FUNCTIONS
    transformers = [
        ClassificationResizer((width, height)),
        transpose_for_pytorch
    ]
    callback_functions = [
        show_image
    ]

    # ANNOTATIONS
    annotation_dir = {
        '0': 'healyth',
        '1': 'multiple_diseases',
        '2': 'rust',
        '3': 'scab'
    }

    predicting_dir = {
        MobileNetV2(
            num_classes, pretrained=True).load_weight(
            './result/fold_0/epoch_1_metric_0.9285714285714286.pth'): 0.928,
        ResNext(
            num_classes, pretrained=True, layer=50).load_weight(
            './result/fold_1/epoch_2_metric_0.9917582417582418.pth'): 0.991,
        ResNext(
            num_classes, pretrained=True, layer=101).load_weight(
            './result/fold_2/epoch_1_metric_0.9423076923076923.pth'): 0.942,
        WideResNet(
            num_classes, pretrained=True, layer=50).load_weight(
            './result/fold_3/epoch_1_metric_0.9725274725274725.pth'): 0.972,
        WideResNet(
            num_classes, pretrained=True, layer=101).load_weight(
            './result/fold_4/epoch_1_metric_0.9642857142857143.pth'): 0.964,
    }
    predicting_table = PredictingTable(predicting_dir)
    annotation_table = AnnotationTable(annotation_dir)

    dataset = ClassificationDataset(
        dataset_maker.test_inputs, dataset_maker.test_inputs, batch_size, transformers=transformers)

    predictor = ClassificationPredictor(predicting_table, annotation_table, dataset, result_path)
    predictor.run()


if __name__ == "__main__":
    main()
