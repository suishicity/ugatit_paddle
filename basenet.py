import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn


def conv_norm(in_ch, out_ch, ksize, stride, padding=0, groups=None, padding_type='reflect',
                param_attr=None, bias_attr=None, norm=None, act=None):
    layers = []
    if padding_type == 'reflect':
        layers.append(ReflectionPad2D(padding))
        padding = 0
    layers.append(
        nn.Conv2D(in_ch, out_ch, ksize, stride, padding, groups=groups, param_attr=param_attr, bias_attr=bias_attr)
    )
    if norm is not None:
        layers.append(norm)
    if act is not None:
        layers.append(act)

    return fluid.dygraph.Sequential(*layers)
    

class ReflectionPad2D(fluid.dygraph.Layer):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, mode='reflect')


class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return fluid.layers.leaky_relu(x, self.alpha)

class ReLU(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return fluid.layers.relu(x)


class Upsample(fluid.dygraph.Layer):
    def __init__(self, out_shape=None, scale=None):
        super().__init__()
        self.out_shape = out_shape
        self.scale = scale

    def forward(self, x):
        return fluid.layers.resize_bilinear(x, out_shape=self.out_shape, scale=self.scale)

    
class SpectralNormConv2D(nn.Conv2D):
    def forward(self, input):
        attrs = ('strides', self._stride, 'paddings', self._padding,
                    'dilations', self._dilation, 'groups', self._groups
                    if self._groups else 1, 'use_cudnn', self._use_cudnn)

        weight = fluid.layers.spectral_norm(self.weight, dim=1)
        out = fluid.dygraph.core.ops.conv2d(input, weight, *attrs)
        pre_bias = out
        pre_act = fluid.dygraph_utils._append_bias_in_dygraph(pre_bias, self.bias, 1)
        return fluid.dygraph_utils._append_activation_in_dygraph(pre_act, self._act)


class SpectralNormLinear(nn.Linear):
    def forward(self, input):
        pre_bias = fluid.framework._varbase_creator(dtype=input.dtype)
        weight = fluid.layers.spectral_norm(self.weight, dim=0)
        fluid.dygraph.core.ops.matmul(input, weight, pre_bias, 'transpose_X', False,
                        'transpose_Y', False, "alpha", 1)
        pre_act = fluid.dygraph_utils._append_bias_in_dygraph(
            pre_bias, self.bias, axis=len(input.shape) - 1)

        return fluid.dygraph_utils._append_activation_in_dygraph(pre_act, self._act)