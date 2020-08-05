import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as nn

from .basenet import conv_norm, ReLU, LeakyReLU, ReflectionPad2D


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
            self.fc = nn.Sequential(
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear((img_size // mult)**2 * ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                nn.Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
            )

        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beata = nn.Linear(ngf * mult, ngf * mult, bias_attr=False)

        # for i in range(n_blocks):


        
        


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
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU

        self.pad2 = ReflectionPad2D(1)
        self.conv2 =nn.Conv2D(dim, dim, 3, 1, 0, bias_attr=bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.conv1(self.pad1(x))
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.conv2(self.pad2(x))
        out = self.norm2(out)

        return out + x


class AdaILN(nn.Layer):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        rho = fluid.layers.fill_constant((1, num_features, 1, 1), dtype='float32', value=0.9)
        self.rho = self.add_parameter('rho', rho)
        
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
        gamma = fluid.layers.reshpe(gamma, (gamma.shape[0], gamma.shape[1], 1, 1))
        beta = fluid.layers.reshpe(beta, (beta.shape[0], beta.shape[1], 1, 1))
        out = out * gamma + beta
        return out