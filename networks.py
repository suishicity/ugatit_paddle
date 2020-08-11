import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as nn
from paddle.fluid.initializer import ConstantInitializer, NormalInitializer

from basenet import (conv_norm, ReLU, LeakyReLU, ReflectionPad2D, Upsample, 
    SpectralNormConv2D, SpectralNormLinear)


class ResnetGenerator(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        down_blocks = [nn.Sequential(
            ReflectionPad2D(3),
            nn.Conv2D(input_nc, ngf, 7, 1, 0,  bias_attr=False),
            nn.InstanceNorm(ngf), 
            ReLU()
        )]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i 
            down_blocks.append(nn.Sequential(
                ReflectionPad2D(1),
                nn.Conv2D(ngf * mult, ngf * mult * 2, 3, 2, bias_attr=False),
                nn.InstanceNorm(ngf * mult * 2),
                ReLU()
            ))

        mult = 2**n_downsampling
        for i in range(n_blocks):
            down_blocks.append(ResnetBlock(ngf * mult, bias=False))

        self.gap_fc = nn.Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.Conv2D(ngf * mult * 2, ngf * mult, 1, 1)
        self.relu = ReLU()

        if self.light:
            self.FC= nn.Sequential(
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
            )
        else:
            self.FC = nn.Sequential(
                nn.Linear((img_size // mult)**2 * ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
            )

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias_attr=False)

        self.UpBlock1 = nn.LayerList()
        for i in range(n_blocks):
            self.UpBlock1.append(ResnetAdaILNBlock(ngf * mult, bias=False))
        
        up_blocks = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_blocks += [
                Upsample(scale=2),
                ReflectionPad2D(1),
                nn.Conv2D(ngf * mult, ngf * mult // 2, 3, 1, 0, bias_attr=False),
                ILN(ngf * mult // 2),
                ReLU(),
            ]
        up_blocks += [
            ReflectionPad2D(3),
            nn.Conv2D(ngf, output_nc, 7, 1, 0, bias_attr=False, act='tanh')
        ]

        self.DownBlock = nn.Sequential(*down_blocks)
        self.UpBlock2 = nn.Sequential(*up_blocks)
        self.avg_pool = nn.Pool2D(pool_type='avg', global_pooling=True)
        self.max_pool = nn.Pool2D(pool_type='max', global_pooling=True)
    

    def forward(self, x):
        x = self.DownBlock(x)

        gap = self.avg_pool(x)
        gap_logits = self.gap_fc(fluid.layers.reshape(gap, (gap.shape[0], -1)))
        gap_weight = fluid.layers.reshape(self.gap_fc.weight, (1, -1, 1, 1))
        gap = x * gap_weight

        gmp = self.max_pool(x)
        gmp_logits = self.gmp_fc(fluid.layers.reshape(gmp, (gmp.shape[0], -1)))
        gmp_weight = fluid.layers.reshape(self.gmp_fc.weight, (1, -1, 1, 1))
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logits, gmp_logits], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = self.avg_pool(x, 1)
            x_ = self.FC(fluid.layers.reshape(x_, (x.shape[0], -1)))
        else:
            x_ = self.FC(fluid.layers.reshape(x, (x.shape[0], -1)))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = self.UpBlock1[i](x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(nn.Layer):
    def __init__(self, dim, bias=None):
        super().__init__()
        layers = [nn.Sequential(
            ReflectionPad2D(1),
            nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias),
            nn.InstanceNorm(dim),
            ReLU()
        )]
        layers.append(nn.Sequential(
            ReflectionPad2D(1),
            nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias),
            nn.InstanceNorm(dim),
        ))
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Layer):
    def __init__(self, dim, bias=None):
        super().__init__()
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias)
        self.norm1 = AdaILN(dim)
        self.relu = ReLU()

        self.pad2 = ReflectionPad2D(1)
        self.conv2 =nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias)
        self.norm2 = AdaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.conv1(self.pad1(x))
        out = self.norm1(out, gamma, beta)
        out = self.relu(out)
        out = self.conv2(self.pad2(x))
        out = self.norm2(out, gamma, beta)

        return out + x


class AdaILN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = self.create_parameter((1, num_features, 1, 1), dtype='float32', 
                        default_initializer=ConstantInitializer(0.9))  
        
    def normalize(self, x, dim):
        x_mean = fluid.layers.reduce_mean(x, dim, keep_dim=True)
        x2 = fluid.layers.square(x)
        x2_mean = fluid.layers.reduce_mean(x2, dim, keep_dim=True)
        x_var = x2_mean - fluid.layers.square(x_mean)
        out = (x - x_mean) / fluid.layers.sqrt(x_var + self.eps)
        return out
    

    def forward(self, x, gamma, beta):
        out_in = self.normalize(x, dim=[2, 3])
        out_ln = self.normalize(x, dim=[1, 2, 3])
        out = self.rho * out_in + (1 - self.rho) * out_ln
        gamma = fluid.layers.reshape(gamma, (gamma.shape[0], gamma.shape[1], 1, 1))
        beta = fluid.layers.reshape(beta, (beta.shape[0], beta.shape[1], 1, 1))
        out = out * gamma + beta
        return out


class ILN(nn.Layer):
    def __init__(self , num_features, eps=1e-5):
        super().__init__()
        size = (1, num_features, 1, 1)
        self.eps = eps
        self.rho = self.create_parameter(size, dtype='float32', default_initializer=ConstantInitializer(0.0))
        self.gamma =self.create_parameter(size, dtype='float32', default_initializer=ConstantInitializer(1.0)) 
        self.beta =self.create_parameter(size, dtype='float32', default_initializer=ConstantInitializer(1.0)) 

    def normalize(self, x, dim):
        x_mean = fluid.layers.reduce_mean(x, dim, keep_dim=True)
        x2 = fluid.layers.square(x)
        x2_mean = fluid.layers.reduce_mean(x2, dim, keep_dim=True)
        x_var = x2_mean - fluid.layers.square(x_mean)
        out = (x - x_mean) / fluid.layers.sqrt(x_var + self.eps)
        return out

    def forward(self, x):
        out_in = self.normalize(x, dim=[2, 3])
        out_ln = self.normalize(x, dim=[1, 2, 3])
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * self.gamma + self.beta
        return out


class Discriminator(nn.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super().__init__()
        layers = [
            ReflectionPad2D(1),
            SpectralNormConv2D(input_nc, ndf, 4, 2, 0, bias_attr=True),
            LeakyReLU(0.2)
        ]
        
        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            layers += [
                ReflectionPad2D(1),
                SpectralNormConv2D(ndf * mult, ndf * mult * 2, 4, 2, 0, bias_attr=True),
                LeakyReLU(0.2),
            ]
        
        mult = 2 ** (n_layers - 3)
        layers += [
            ReflectionPad2D(1),
            SpectralNormConv2D(ndf * mult, ndf * mult * 2, 4, 1, 0, bias_attr=True),
            LeakyReLU(0.2),
        ]

        mult = 2 ** (n_layers - 2)
        self.gap_fc = SpectralNormLinear(ndf * mult, 1, bias_attr=False)
        self.gmp_fc = SpectralNormLinear(ndf * mult, 1, bias_attr=False)
        self.conv1x1 = nn.Conv2D(ndf * mult * 2, ndf * mult, 1, 1)
        self.leaky_relu = LeakyReLU(0.2)
        self.pad = ReflectionPad2D(1)
        self.conv = SpectralNormConv2D(ndf * mult, 1, 4, 1, bias_attr=False)
        self.model = nn.Sequential(*layers)
        self.avg_pool = nn.Pool2D(pool_type='avg', global_pooling=True)
        self.max_pool = nn.Pool2D(pool_type='max', global_pooling=True)

    def forward(self, input):
        x = self.model(input)

        gap = self.avg_pool(x)
        gap_logits = self.gap_fc(fluid.layers.reshape(gap, (gap.shape[0], -1)))
        gap_weight = fluid.layers.reshape(self.gap_fc.weight, (1, -1, 1, 1))
        gap = x * gap_weight

        gmp = self.max_pool(x)
        gmp_logits = self.gmp_fc(fluid.layers.reshape(gmp, (gmp.shape[0], -1)))
        gmp_weight = fluid.layers.reshape(self.gmp_fc.weight, (1, -1, 1, 1))
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logits, gmp_logits], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class GANLoss(fluid.dygraph.Layer):
    def __init__(self, gan_mode):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = fluid.dygraph.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = fluid.dygraph.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return fluid.layers.ones_like(prediction)
        else:
            return fluid.layers.zeros_like(prediction)

    def forward(self, prediction, target_is_real):
        if self.gan_mode == 'lsgan':
            target = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target)
        elif self.gan_mode == 'vanilla':
            target = self.get_target_tensor(prediction, target_is_real)
            prediction = fluid.layers.sigmoid(prediction)
            loss = self.loss(prediction, target)
        elif self.gan_mode == 'wgangp':
            loss = -fluid.layers.mean(prediction) if target_is_real else fluid.layers.mean(prediction)

        return loss


def clip_rho(net, vmin=0, vmax=1):
    for name, param in net.named_parameters():
        if 'rho' in name:
            param.set_value(fluid.layers.clip(param, vmin, vmax))

