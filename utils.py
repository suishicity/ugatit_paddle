import cv2
import numpy as np


def normalize(img, mean=0.5, std=0.5):
    img = (img / 255.0 - mean) / std
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    return img


def denormalize(img, mean=0.5, std=0.5, transpose=False, out_format='RGB'):
    if transpose:
        img = np.transpose(img, (1, 2, 0))
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype('uint8')
    if out_format == 'BGR':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
        

def make_grid(imgs, ncol=-1, padding=1, pad_value=0):
    h, w = imgs[0].shape[:2]
    if ncol < 0:
        ncol = int(np.sqrt(len(imgs)))
    nrow = int(np.ceil(len(imgs) / ncol))
    H = h * nrow + padding * (nrow - 1)
    W = w * ncol + padding * (ncol - 1)
    if imgs[0].ndim == 2:
        result = np.full((H, W), pad_value, dtype=imgs[0].dtype)
    else:
        result = np.full((H, W, imgs[0].shape[-1]), pad_value, dtype=imgs[0].dtype)
    for k in range(len(imgs)):
        i = k // ncol
        j = k - i * ncol
        result[i*(h+padding): i*(h+padding) + h, j*(w+padding): j*(w+padding) + w] = imgs[k]
    result = np.squeeze(result)
    return result 


def gen_cam(x, out_size=(256, 256), out_format='RGB'):
    cam = (x - x.min()) / (x.max() - x.min())
    cam = (x * 255).astype('uint8')
    cam = cv2.resize(cam, out_size, interpolation=cv2.INTER_LINEAR)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    if out_format == 'RGB':
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    return cam
