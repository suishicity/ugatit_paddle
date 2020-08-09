import cv2
import numpy as np


def normalize(img, mean=0.5, std=0.5):
    img = (img / 255.0 - mean) / std
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    return img


def denormalize(img, mean=0.5, std=0.5, transpose=True):
    if transpose:
        img = np.transpose(img, (1, 2, 0))
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def image_transform(mode='train', img_size=(256, 256)):
    new_size = (img_size[0] + 30, img_size[1] + 30)

    def aug(img):
        if mode == 'train':
            if np.random.uniform() < 0.5:
                img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)
                h = np.random.randint(0, 30)
                w = np.random.randint(0, 30)
                img = np.ascontiguousarray(img[h: h + img_size[0], w: w + img_size[1]])
            if np.random.uniform() < 0.5:
                img = cv2.flip(img, 1)

        img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
        img = normalize(img)
        return img

    return aug
        
