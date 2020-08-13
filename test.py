import os
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pylab as plt

from networks import ResnetGenerator, Discriminator, clip_rho


def test_net():
    fluid.dygraph.enable_dygraph(fluid.CPUPlace())
    x = fluid.layers.uniform_random((1, 3, 128, 128))
    netG = ResnetGenerator(3, 3, img_size=128)
    clip_rho(netG)

    for name, param in netG.named_parameters():
        if 'rho' in name:
            print(name, param.numpy().max())

def test_data():
    from dataset import ImageFolder
    from utils import image_transform, denormalize

    data_dir = r"F:\Data\selfi2anime"
    trainA = ImageFolder(os.path.join(data_dir, 'trainA'), image_transform())

    plt.ion()
    for i in range(100):
        x = trainA[i]
        img = denormalize(x, transpose=True)
        plt.imshow(img)
        plt.waitforbuttonpress()


if __name__ == "__main__":
    test_data()    





