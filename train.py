import os

import torch
from torch import nn
from torch import optim
import dataloader

import models

class Trainer:
    def __init__(self, Config):
        self.__init_generator()
        self.__init_discriminator()
        self.fixed_noise = torch.randn(
            64, Config.N_DIMS, 1, 1, device=Config.DEVICE)

        self.real_label = 1
        self.fake_label = 1

        self.stepG = optim.Adam(
            self.netG.parameters(), lr=Config.LEARNING_RATE,
            betas=(Config.BETA, 0.999))
        self.stepD = optim.Adam(
            self.netD.parameters(), lr=Config.LEARNING_RATE,
            betas=(Config.BETA, 0.999))

        self.criterion = nn.BCELoss()
        self.train_params = {
            'img_list': [],
            'generator_losses': [],
            'discriminator_losses': []
        }
        self.current_iter = {
            'D_x': 0,
            'errD': 0,
            'errG': 0,
            'D_G_z1': 0,
            'D_G_z2': 0
        }
        self.Config = Config

    def __init_generator(self, Config):
        self.netG = models.Generator(
            Config.N_DIMS, Config.GEN_FEATURE_MAPS, Config.N_CHANNELS)
        self.netG.to(Config.DEVICE)
        self.netG.apply(models.weights_init)

    def __init_discriminator(self, Config):
        self.netD = models.Discriminator(
            Config.N_CHANNELS, Config.DIS_FEATURE_MAPS)
        self.netD.to(Config.DEVICE)
        self.netD.apply(models.weights_init)

    def __train_on_real(self, real_imgs):
        self.netD.zero_grad()
        label = torch.full(
            (real_imgs.size(0),), self.real_label, device=self.Config.DEVICE)
        out = self.netD(real_imgs).view(-1)
        err_real_imgs = self.criterion(out, label)
        err_real_imgs.backward()
        self.current_iter['D_x'] = out.mean().item()
        self.current_iter['err_real_imgs'] = err_real_imgs.item()

    def __train_on_fake(self):
        noise = torch.randn(
            self.Config.BATCH_SIZE, self.Config.N_DIMS,
            1, 1, device=self.Config.DEVICE)
        fake = self.netG(noise)
        label = torch.full((
            self.Config.BATCH_SIZE,), self.fake_label, device=self.Config.DEVICE)
        out = self.netD(fake).view(-1)
        err_fake_imgs = self.criterion(out, label)
        err_fake_imgs.backward()

        self.current_iter['D_G_z1'] = out.mean().item()
        self.current_iter['err_fake_imgs'] = err_fake_imgs

        self.stepD.step()

        self.netG.zero_grad()
        label.fill_(self.real_label)
        out = self.netD(fake).view(-1)
        errG = self.criterion(out, label)
        errG.backward()
        self.current_iter['D_G_z2'] = out.mean().item()
        self.current_iter['errG'] = errG.item()
        self.stepG.step()

    def __print_iteration(self, nth_epoch, batch_num, n_batches):
        s = '[{}/{}][{}/{}]\tLoss_D: {:.4f}\t'
        'Loss_G: {:.4f}\tD(x): {:.4f}\t'
        'D(G(z)): {:.4f} / {:.4f}'
        print(s.format(
            nth_epoch, self.Config.EPOCHS, batch_num, n_batches,
            self.current_iter['errD'], self.current_iter['errG'],
            self.current_iter['D_x'], self.current_iter['D_G_z1'],
            self.current_iter['D_G_z1']))

    def train(self):
        self.img_list = []
        self.losses = {
            'G': [],
            'D': []
        }
        it_ctr = 0

        data = dataloader.CelebFacesDataloaders(
            self.Config.DATA_ROOT, self.Config.IMG_SIZE,
            self.Config.BATCH_SIZE, self.Config.NUM_WORKERS)

        print('Starting the training...')
        for epoch in range(self.Config.EPOCHS):
            for i, data in enumerate(data, 0):

                real_imgs, _ = data
                real_imgs = real_imgs.to(self.Config.DEVICE)
                self.__train_on_real(real_imgs)
                self.__train_on_fake()

                if it_ctr % self.Config.LOG_STEP == 0:
                    self.__print_iteration(epoch, i, len(data))

                if ((it_ctr % self.Config.IMG_LOG_STEP == 0) or
                    ((epoch == self.Config.EPOCHS - 1) and (i == len(data)-1))):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(fake)
                    torch.save(self.netG.state_dict,
                        os.path.join(self.Config.model_path, 'generator.model'))
                    torch.save(self.netD.state_dict,
                        os.path.join(self.Config.model_path, 'discriminator.model'))

                it_ctr += 1
