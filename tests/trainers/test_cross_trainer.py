import unittest
import numpy as np

from blues.models.object_detections.efficientdet.efficientdet import EfficientDetector
from blues.datasets.object_detection_dataset import ObjectDetectionDataset
from blues.trainers.xtrainer import XTrainer


class TestCrossTrainer(unittest.TestCase):

    def test_cross_trainer(self):
        batch_size = 10
        width = 512
        height = 512

        model = EfficientDetector(5)
        dummy_inputs = []
        dummy_teachers = []
        for i in range(100):
            dummy_inputs.append(np.random.rand(3, width, height))
            dummy_teachers.append(np.random.rand(12, 5) * 10)
        dataset = ObjectDetectionDataset(dummy_inputs, dummy_teachers, batch_size)

        metric = lambda x, y: 1
        trainer = XTrainer(model, dataset, 2, 5, './hiphop', metric)
        trainer.run()


if __name__ == "__main__":
    unittest.main()
