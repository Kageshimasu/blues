import random


class DataAugmentor:

    def __init__(self, data_augmentations):
        self._data_augmentations = data_augmentations

    def __len__(self):
        return len(self._data_augmentations)

    def __call__(self, inputs, teachers):
        prob = len(self) + 1
        for augment_function in self._data_augmentations:
            if random.random() < prob:
                inputs, teachers = augment_function(inputs, teachers)
        return inputs, teachers
