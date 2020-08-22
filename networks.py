import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as nn
from paddle.fluid.initializer import ConstantInitializer, NormalInitializer

from basenet import (conv_norm, ReLU, LeakyReLU, ReflectionPad2D, Upsample, 
    SpectralNormConv2D, SpectralNormLinear)

# param_attr = fluid.ParamAttr(initializer=NormalInitializer(0, 0.02, seed=0))
param_attr = None


class ResnetGenerator(nn.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.light = light
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        self.img_size = img_size

        down_blocks = [nn.Sequential(
            ReflectionPad2D(3),
            nn.Conv2D(input_nc, ngf, 7, 1, 0,  bias_attr=False, param_attr=param_attr),
            nn.InstanceNorm(ngf), 
            ReLU()
        )]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i 
            down_blocks.append(nn.Sequential(
                ReflectionPad2D(1),
                nn.Conv2D(ngf * mult, ngf * mult * 2, 3, 2, bias_attr=False, param_attr=param_attr),
                nn.InstanceNorm(ngf * mult * 2),
                ReLU()
            ))

        mult = 2**n_downsampling
        for i in range(n_blocks):
            down_blocks.append(ResnetBlock(ngf * mult, bias=False, param_attr=param_attr))

        self.gap_fc = nn.Linear(ngf * mult, 1, bias_attr=False, param_attr=param_attr)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias_attr=False, param_attr=param_attr)
        self.conv1x1 = nn.Conv2D(ngf * mult * 2, ngf * mult, 1, 1, param_attr=param_attr)
        self.relu = ReLU()

        if self.light:
            self.FC= nn.Sequential(
                nn.Linear(ngf * mult*16, ngf * mult, bias_attr=False, act='relu', param_attr=param_attr),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu', param_attr=param_attr),
            )
        else:
            self.FC = nn.Sequential(
                nn.Linear((img_size[0] // mult) * (img_size[1] // mult) * ngf * mult, ngf * mult, 
                          bias_attr=False, act='relu', param_attr=param_attr),
                nn.Linear(ngf * mult, ngf * mult, param_attr=param_attr, bias_attr=False, act='relu'),
            )

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias_attr=False, param_attr=param_attr)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias_attr=False, param_attr=param_attr)

        self.UpBlock1 = nn.LayerList()
        for i in range(n_blocks):
            self.UpBlock1.append(ResnetAdaILNBlock(ngf * mult, bias=False))
        
        up_blocks = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up_blocks += [
                Upsample(scale=2),
                ReflectionPad2D(1),
                nn.Conv2D(ngf * mult, ngf * mult // 2, 3, 1, 0, bias_attr=False, param_attr=param_attr),
                ILN(ngf * mult // 2),
                ReLU(),
            ]
        up_blocks += [
            ReflectionPad2D(3),
            nn.Conv2D(ngf, output_nc, 7, 1, 0, bias_attr=False, param_attr=param_attr, act='tanh')
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
        g = fluid.layers.concat([gap, gmp], 1)
        g = self.relu(self.conv1x1(g))

        heatmap = fluid.layers.reduce_sum(g, dim=1, keep_dim=True)

        if self.light:
            # x_ = self.avg_pool(x)
            x_ = fluid.layers.adaptive_pool2d(x, [4, 4], 'avg')
            x_ = self.FC(fluid.layers.reshape(x_, (x.shape[0], -1)))
        else:
            x_ = self.FC(fluid.layers.reshape(x, (x.shape[0], -1)))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = self.UpBlock1[i](x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(nn.Layer):
    def __init__(self, dim, bias=None, param_attr=None):
        super().__init__()
        layers = [nn.Sequential(
            ReflectionPad2D(1),
            nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias, param_attr=param_attr),
            nn.InstanceNorm(dim),
            ReLU()
        )]
        layers.append(nn.Sequential(
            ReflectionPad2D(1),
            nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias, param_attr=param_attr),
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
        self.conv1 = nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias, param_attr=param_attr)
        self.norm1 = AdaILN(dim)
        self.relu = ReLU()

        self.pad2 = ReflectionPad2D(1)
        self.conv2 =nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias, param_attr=param_attr)
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
        self.gamma = self.create_parameter(size, dtype='float32', default_initializer=ConstantInitializer(1.0)) 
        self.beta = self.create_parameter(size, dtype='float32', default_initializer=ConstantInitializer(0.0)) 

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
        # x = fluid.layers.concat([gap, gmp], 1)
        # x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class GANLoss(fluid.dygraph.Layer):
    def __init__(self, gan_mode):
        super().__init__()
        assert gan_mode in ['lsgan', 'vanilla', 'wgangp', 'hinge']
        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss_fn = nn.BCELoss()

    def forward(self, fake, real=None):
        if real is None:
            if self.gan_mode == 'lsgan':
                loss = fluid.layers.mean(fluid.layers.square(fake - 1))
            elif self.gan_mode == 'vanilla':
                fake_sigmoid = fluid.layers.sigmoid(fake)
                loss = self.loss_fn(fake_sigmoid, fluid.layers.ones_like(fake))
            elif self.gan_mode == 'wgangp' or self.gan_mode == 'hinge':
                loss = - fluid.layers.mean(fake)
        else:
            if self.gan_mode == 'lsgan':
                loss = fluid.layers.mean(fluid.layers.square(real - 1)) \
                     + fluid.layers.mean(fluid.layers.square(fake))
            elif self.gan_mode == 'vanilla':
                real_sigmoid = fluid.layers.sigmoid(real)
                fake_sigmoid = fluid.layers.sigmoid(fake)
                loss = self.loss_fn(real_sigmoid, fluid.layers.ones_like(real)) \
                     + self.loss_fn(fake_sigmoid, fluid.layers.zeros_like(fake))
 
            elif self.gan_mode == 'wgangp':
                loss = -fluid.layers.mean(real) + fluid.layers.mean(fake)
            elif self.gan_mode == 'hinge':
                loss = fluid.layers.mean(fluid.layers.relu(1 - real)) \
                     + fluid.layers.mean(fluid.layers.relu(1 + fake))

        return loss


def clip_rho(net, vmin=0, vmax=1):
    for name, param in net.named_parameters():
        if 'rho' in name:
            param.set_value(fluid.layers.clip(param, vmin, vmax))

