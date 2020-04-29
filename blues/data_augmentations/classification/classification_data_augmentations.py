import torchvision


def random_rotation(inputs, teachers, degrees=(-20, 20)):
    return torchvision.transforms.RandomRotation(degrees)(inputs), teachers
