import cv2
import numpy as np
import imgaug.augmenters as iaa


img = cv2.imread('test.jpg') / 255
print(img.dtype)
batch = 10
images = np.zeros((batch, img.shape[0], img.shape[1], 3), np.float32)
for i in range(batch):
    images[i] = img.copy()

seq = iaa.Sequential([
    iaa.Crop(),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0)),
    iaa.Cutout(),
    iaa.Multiply()
    # iaa.Fliplr(),
])

images_aug = seq(images=images)

for i in range(batch):
    cv2.imshow('test', images_aug[i])
    cv2.imshow('original', images[i])
    cv2.waitKey(1000)
