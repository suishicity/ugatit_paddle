import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
import matplotlib.pylab as plt

from networks import ResnetGenerator, Discriminator, clip_rho


def test_net():
    fluid.dygraph.enable_dygraph(fluid.CPUPlace())
    np.random.seed(0)
    x = np.random.uniform(0, 1, size=(1, 3, 128, 128)).astype('float32')
    x = to_variable(x)
    netG = ResnetGenerator(3, 3, img_size=128)
    netG.eval()
    with fluid.dygraph.no_grad():
        y = netG(x)[0].numpy()
    print(y.sum())
    # sd = netG.state_dict()
    # print(len(sd))
    # for key in sd.keys():
    #     print(key)

def test_pool():
    fluid.dygraph.enable_dygraph(fluid.CPUPlace())
    x = fluid.layers.uniform_random((1, 3, 32, 32))
    y = fluid.layers.adaptive_pool2d(x, [2, 2])
    print(y.shape)



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
    # test_data()    

    # test_net()

    test_pool()





