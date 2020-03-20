import os

import torch
from torch import nn
from torch import optim
import dataloader

import models

class Trainer:
    def __init__(self, Config):
        self.Config = Config
        self.__init_generator()
        self.__init_discriminator()

        self.__input = torch.FloatTensor(
            self.Config.BATCH_SIZE, 3, self.Config.IMG_SIZE, self.Config.IMG_SIZE)
        self.__noise = torch.FloatTensor(
            self.Config.BATCH_SIZE, self.Config.N_DIMS, 1, 1)
        self.__fixed_noise = torch.FloatTensor(
            self.Config.BATCH_SIZE, self.Config.N_DIMS, 1, 1).normal_(0, 1)
        self.__label = torch.FloatTensor(self.Config.BATCH_SIZE)

        self.__input = self.__input.to(self.Config.DEVICE)
        self.__noise = self.__noise.to(self.Config.DEVICE)
        self.__fixed_noise = self.__fixed_noise.to(self.Config.DEVICE)
        self.__label = self.__label.to(self.Config.DEVICE)

        self.real_label = 1
        self.fake_label = 0

        self.stepG = optim.Adam(
            self.netG.parameters(), lr=Config.LEARNING_RATE,
            betas=(Config.BETA, 0.999))
        self.stepD = optim.Adam(
            self.netD.parameters(), lr=Config.LEARNING_RATE,
            betas=(Config.BETA, 0.999))

        self.criterion = nn.BCELoss()
        self.criterion.to(self.Config.DEVICE)

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

    def __init_generator(self):
        self.netG = models.Generator(
            self.Config.N_DIMS, self.Config.GEN_FEATURE_MAPS, self.Config.N_CHANNELS)
        self.netG.to(self.Config.DEVICE)
        self.netG.apply(models.weights_init)

    def __init_discriminator(self):
        self.netD = models.Discriminator(
            self.Config.N_CHANNELS, self.Config.DIS_FEATURE_MAPS)
        self.netD.to(self.Config.DEVICE)
        self.netD.apply(models.weights_init)

    def __train_on_real(self, real_imgs):
        self.netD.zero_grad()
        self.__input.resize_as_(real_imgs).copy_(real_imgs)
        self.__label.resize_(real_imgs.size(0)).fill_(self.real_label)

        input_var = torch.autograd.Variable(self.__input)
        label_var = torch.autograd.Variable(self.__label)

        out = self.netD(input_var).view(-1)
        err_real_imgs = self.criterion(out, label_var)
        err_real_imgs.backward()
        self.stepD.step()

        self.current_iter['D_x'] = out.mean().item()
        self.current_iter['err_real_imgs'] = err_real_imgs.item()

    def __train_on_fake(self):
        self.__noise.resize_(
            self.Config.BATCH_SIZE, self.Config.N_DIMS, 1, 1).normal_(0, 1)
        noise_var = torch.autograd.Variable(self.__noise)
        fake = self.netG(noise_var)
        self.__label.resize_(self.Config.BATCH_SIZE).fill_(self.fake_label)
        label_var = torch.autograd.Variable(self.__label)
        out = self.netD(fake.detach()).view(-1)
        err_fake_imgs = self.criterion(out, label_var)
        err_fake_imgs.backward()

        self.current_iter['D_G_z1'] = out.mean().item()
        self.current_iter['err_fake_imgs'] = err_fake_imgs.item()

        self.stepD.step()

        self.netG.zero_grad()
        
        label_var = torch.autograd.Variable(self.__label.fill_(self.real_label))
        out = self.netD(fake).view(-1)
        errG = self.criterion(out, label_var)
        errG.backward()
        
        self.current_iter['D_G_z2'] = out.mean().item()
        self.current_iter['errG'] = errG.item()
        self.stepG.step()

    def __print_iteration(self, nth_epoch, batch_num, n_batches):
        s = '[{}/{}][{}/{}]\tLoss_D: {:.4f}\t' + \
            'Loss_G: {:.4f}\tD(x): {:.4f}\t' + \
            'D(G(z)): {:.4f} / {:.4f}'
        print(s.format(
            nth_epoch, self.Config.EPOCHS, batch_num, n_batches,
            self.current_iter['errD'], self.current_iter['errG'],
            self.current_iter['D_x'], self.current_iter['D_G_z1'],
            self.current_iter['D_G_z2']))

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
            for i, batch in enumerate(data.dataloader, 0):

                real_imgs, _ = batch
                real_imgs = real_imgs.to(self.Config.DEVICE)
                self.__train_on_real(real_imgs)
                self.__train_on_fake()
                self.current_iter['errD'] = self.current_iter['err_fake_imgs'] + \
                    self.current_iter['err_real_imgs']

                if it_ctr % self.Config.LOG_STEP == 0:
                    self.__print_iteration(epoch, i, len(data.dataloader))

                if ((it_ctr % self.Config.IMG_LOG_STEP == 0) or
                    ((epoch == self.Config.EPOCHS - 1) and (i == len(data.dataloader)-1))):
                    with torch.no_grad():
                        fake = self.netG(self.__fixed_noise).detach().cpu()
                    self.img_list.append(fake)
                    torch.save(self.netG.state_dict,
                        os.path.join(self.Config.MODEL_PATH, 'generator.model'))
                    torch.save(self.netD.state_dict,
                        os.path.join(self.Config.MODEL_PATH, 'discriminator.model'))

                it_ctr += 1
