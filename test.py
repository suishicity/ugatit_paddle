import os
import numpy as np
import paddle
import paddle.fluid as fluid

from networks import ResnetGenerator, Discriminator, clip_rho


def test_net():
    fluid.dygraph.enable_dygraph(fluid.CPUPlace())
    x = fluid.layers.uniform_random((1, 3, 128, 128))

    netG = ResnetGenerator(3, 3, img_size=128)

    # for name, module in netG.named_parameters():
        # print(name, module.__class__.__name__)
    clip_rho(netG)

    for name, param in netG.named_parameters():
        if 'rho' in name:
            print(name, param.numpy().max())




if __name__ == "__main__":
    test_net()    





