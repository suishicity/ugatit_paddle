import os
import yaml
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import nn
from paddle.fluid.dygraph import to_variable
import numpy as np

from trainer import DefaultTrainer
from networks import ResnetGenerator, Discriminator, GANLoss, clip_rho
from scheduler import LinearScheduler


class UGATIT(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.GAN_loss = GANLoss(cfg.gan_mode)
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCELoss()

        self.lambda_adv = cfg.lambda_adv
        self.lambda_cyc = cfg.lambda_cyc
        self.lambda_idt = cfg.lambda_idt
        self.lambda_cam = cfg.lambda_cam

    def build_models(self, cfg):
        width = cfg.base_width
        self.genAB = ResnetGenerator(3, 3, ngf=width, n_blocks=cfg.n_res, img_size=cfg.img_size, light=cfg.light)
        self.genBA = ResnetGenerator(3, 3, ngf=width, n_blocks=cfg.n_res, img_size=cfg.img_size, light=cfg.light)
        self.disGA = Discriminator(3, width, 7)
        self.disGB = Discriminator(3, width, 7)
        self.disLA = Discriminator(3, width, 5)
        self.disLB = Discriminator(3, width, 5)

        schedulerG = LinearScheduler(cfg.lr, cfg.iterations, cfg.decay_step)
        schedulerD = LinearScheduler(cfg.lr, cfg.iterations, cfg.decay_step)
        netG_param = self.genAB.parameters() + self.genBA.parameters()
        netD_param = self.disGA.parameters() + self.disGB.parameters() + \
                     self.disLA.parameters() + self.disLB.parameters()

        self.optimizerG = fluid.optimizer.Adam(schedulerG, 0.5, 0.999, parameter_list=netG_param, 
                            regularization=fluid.regularizer.L2Decay(cfg.weight_decay)) 
        self.optimizerD = fluid.optimizer.Adam(schedulerD, 0.5, 0.999, parameter_list=netD_param,
                            regularization=fluid.regularizer.L2Decay(cfg.weight_decay)) 

        self.model_dict = {
            'genAB': self.genAB,
            'genBA': self.genBA,
            'disGA': self.disGA,
            'disGB': self.disGB,
            'disLA': self.disLA,
            'disLB': self.disLB
        } 
        self.optimizers = [self.optimizerG, self.optimizerG] 

    def step(self, inputs):
        real_A = to_variable(np.asarray(inputs['A'], dtype='float32'))
        real_B = to_variable(np.asarray(inputs['A'], dtype='float32'))

        # update D
        self.optimizerD.clear_gradients()

        fake_A2B, _, _ = self.genAB(real_A)
        fake_B2A, _, _ = self.genBA(real_B)

        real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
        real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
        real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
        real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

        D_ad_loss_GA = self.GAN_loss(real_GA_logit, True) + self.GAN_loss(fake_GA_logit, False)
        D_ad_cam_loss_GA = self.GAN_loss(real_GA_cam_logit, True) + self.GAN_loss(fake_GA_cam_logit, False)
        D_ad_loss_LA = self.GAN_loss(real_LA_logit, True) + self.GAN_loss(fake_LA_logit, False)
        D_ad_cam_loss_LA = self.GAN_loss(real_LA_cam_logit, True) + self.GAN_loss(fake_LA_cam_logit, False)
        D_ad_loss_GB = self.GAN_loss(real_GB_logit, True) + self.GAN_loss(fake_GB_logit, False)
        D_ad_cam_loss_GB = self.GAN_loss(real_GB_cam_logit, True) + self.GAN_loss(fake_GB_cam_logit, False)
        D_ad_loss_LB = self.GAN_loss(real_LB_logit, True) + self.GAN_loss(fake_LB_logit, False)
        D_ad_cam_loss_LB = self.GAN_loss(real_LB_cam_logit, True) + self.GAN_loss(fake_LB_cam_logit, False)

        D_loss_A = self.lambda_adv * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
        D_loss_B = self.lambda_adv * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

        D_loss = D_loss_A + D_loss_B
        D_loss.backward()
        self.optimizerD.minimize(D_loss)

        # update G
        self.optimizerG.clear_gradients()

        fake_A2B, fake_A2B_cam_logit, _ = self.genAB(real_A)
        fake_B2A, fake_B2A_cam_logit, _ = self.genBA(real_B)

        fake_A2B2A, _, _ = self.genBA(fake_A2B)
        fake_B2A2B, _, _ = self.genAB(fake_B2A)

        fake_A2A, fake_A2A_cam_logit, _ = self.genBA(real_A)
        fake_B2B, fake_B2B_cam_logit, _ = self.genAB(real_B)

        fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)       

        G_ad_loss_GA = self.GAN_loss(fake_GA_logit, True)
        G_ad_cam_loss_GA = self.GAN_loss(fake_GA_cam_logit, True)
        G_ad_loss_LA = self.GAN_loss(fake_LA_logit, True)
        G_ad_cam_loss_LA = self.GAN_loss(fake_LA_cam_logit, True)
        G_ad_loss_GB = self.GAN_loss(fake_GB_logit, True)
        G_ad_cam_loss_GB = self.GAN_loss(fake_GB_cam_logit, True)
        G_ad_loss_LB = self.GAN_loss(fake_LB_logit, True)
        G_ad_cam_loss_LB = self.GAN_loss(fake_LB_cam_logit, True)
