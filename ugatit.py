import os
import yaml
import cv2
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import nn
from paddle.fluid.dygraph import to_variable
import numpy as np

from trainer import DefaultTrainer
from networks import ResnetGenerator, Discriminator, GANLoss, clip_rho
from scheduler import LinearScheduler
from utils import make_grid, denormalize, gen_cam


class UGATIT(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.GAN_loss = GANLoss(cfg.gan_mode)
        self.L1_loss = nn.L1Loss()
        self.MSE_loss = nn.MSELoss()
        self.BCE_loss = nn.BCELoss()
        self.Vanilla_loss = GANLoss('vanilla')

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

        D_ad_loss_GA = self.GAN_loss(fake_GA_logit, real_GA_logit)
        D_ad_cam_loss_GA = self.GAN_loss(fake_GA_cam_logit, real_GA_cam_logit)
        D_ad_loss_LA = self.GAN_loss(fake_LA_logit, real_LA_logit)
        D_ad_cam_loss_LA = self.GAN_loss(fake_LA_cam_logit, real_LA_cam_logit)
        D_ad_loss_GB = self.GAN_loss(fake_GB_logit, real_GB_logit)
        D_ad_cam_loss_GB = self.GAN_loss(fake_GB_cam_logit, real_GB_cam_logit)
        D_ad_loss_LB = self.GAN_loss(fake_LB_logit, real_LB_logit)
        D_ad_cam_loss_LB = self.GAN_loss(fake_LB_cam_logit, real_LB_cam_logit)

        # D_ad_loss_GA = self.GAN_loss(real_GA_logit, True) + self.GAN_loss(fake_GA_logit, False)
        # D_ad_cam_loss_GA = self.GAN_loss(real_GA_cam_logit, True) + self.GAN_loss(fake_GA_cam_logit, False)
        # D_ad_loss_LA = self.GAN_loss(real_LA_logit, True) + self.GAN_loss(fake_LA_logit, False)
        # D_ad_cam_loss_LA = self.GAN_loss(real_LA_cam_logit, True) + self.GAN_loss(fake_LA_cam_logit, False)
        # D_ad_loss_GB = self.GAN_loss(real_GB_logit, True) + self.GAN_loss(fake_GB_logit, False)
        # D_ad_cam_loss_GB = self.GAN_loss(real_GB_cam_logit, True) + self.GAN_loss(fake_GB_cam_logit, False)
        # D_ad_loss_LB = self.GAN_loss(real_LB_logit, True) + self.GAN_loss(fake_LB_logit, False)
        # D_ad_cam_loss_LB = self.GAN_loss(real_LB_cam_logit, True) + self.GAN_loss(fake_LB_cam_logit, False)

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

        G_ad_loss_GA = self.GAN_loss(fake_GA_logit)
        G_ad_cam_loss_GA = self.GAN_loss(fake_GA_cam_logit)
        G_ad_loss_LA = self.GAN_loss(fake_LA_logit)
        G_ad_cam_loss_LA = self.GAN_loss(fake_LA_cam_logit)
        G_ad_loss_GB = self.GAN_loss(fake_GB_logit)
        G_ad_cam_loss_GB = self.GAN_loss(fake_GB_cam_logit)
        G_ad_loss_LB = self.GAN_loss(fake_LB_logit)
        G_ad_cam_loss_LB = self.GAN_loss(fake_LB_cam_logit)

        G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
        G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

        G_idt_loss_A = self.L1_loss(fake_A2A, real_A)
        G_idt_loss_B = self.L1_loss(fake_B2B, real_B)

        G_cam_loss_A = self.Vanilla_loss(fake_B2A_cam_logit, fake_A2A_cam_logit)
        G_cam_loss_B = self.Vanilla_loss(fake_A2B_cam_logit, fake_B2B_cam_logit)

        # G_cam_loss_A = self.Vanilla_loss(fake_B2A_cam_logit, True) + self.Vanilla_loss(fake_A2A_cam_logit, False)
        # G_cam_loss_B = self.Vanilla_loss(fake_A2B_cam_logit, True) + self.Vanilla_loss(fake_B2B_cam_logit, False)

        G_loss_A = self.lambda_adv * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                   self.lambda_cyc * G_recon_loss_A + self.lambda_idt * G_idt_loss_A + self.lambda_cam * G_cam_loss_A
        G_loss_B = self.lambda_adv * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                    self.lambda_cyc * G_recon_loss_B + self.lambda_idt * G_idt_loss_B +  self.lambda_cam * G_cam_loss_B

        G_loss = G_loss_A + G_loss_B
        G_loss.backward()
        self.optimizerG.minimize(G_loss)

        clip_rho(self.genAB)
        clip_rho(self.genBA)

        losses = {
            "G_loss": G_loss.numpy(),
            "D_loss": D_loss.numpy(),
            "G_ad_loss_GA": G_ad_loss_GA.numpy(),
            "G_ad_loss_GB": G_ad_loss_GB.numpy(),
            "G_ad_loss_GA": G_ad_loss_GA.numpy(),
            "G_ad_loss_GB": G_ad_loss_GB.numpy(),
            "lr": self.optimizerG.current_step_lr(),
        }

        return losses

    @fluid.dygraph.no_grad()
    def sample_images(self, epoch, inputs):
        self.genAB.eval()
        self.genBA.eval()

        img_A = np.asarray(inputs['A'])
        img_B = np.asarray(inputs['B'])

        real_A = to_variable(np.asarray(inputs['A'], dtype='float32'))
        real_B = to_variable(np.asarray(inputs['B'], dtype='float32'))

        fake_A2B, _, fake_A2B_heatmap = self.genAB(real_A)
        fake_B2A, _, fake_B2A_heatmap = self.genBA(real_B)

        fake_A2B2A, _, fake_A2B2A_heatmap = self.genBA(fake_A2B)
        fake_B2A2B, _, fake_B2A2B_heatmap = self.genAB(fake_B2A)

        img_A = denormalize(img_A[0], transpose=True)
        fake_A2B = denormalize(fake_A2B[0].numpy(), transpose=True)
        fake_A2B_heatmap = gen_cam(fake_A2B_heatmap[0][0].numpy())
        fake_A2B2A = denormalize(fake_A2B2A[0].numpy(), transpose=True)
        fake_A2B2A_heatmap = gen_cam(fake_A2B2A_heatmap[0][0].numpy())

        img_B = denormalize(img_B[0], transpose=True)
        fake_B2A = denormalize(fake_B2A[0].numpy(), transpose=True)
        fake_B2A_heatmap = gen_cam(fake_B2A_heatmap[0][0].numpy())
        fake_B2A2B = denormalize(fake_B2A2B[0].numpy(), transpose=True)
        fake_B2A2B_heatmap = gen_cam(fake_B2A2B_heatmap[0][0].numpy())        

        grid = make_grid([
            img_A, fake_A2B_heatmap, fake_A2B, fake_A2B2A_heatmap, fake_A2B2A,
            img_B, fake_B2A_heatmap, fake_B2A, fake_B2A2B_heatmap, fake_B2A2B 
        ], ncol=5, pad_value=255)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.output_image_dir, '{}.png'.format(epoch)), grid)