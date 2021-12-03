import os
import networks
import torch
import loss
import torch.nn as nn
import numpy as np

from models.model import Model

class AttGAN(Model):
    def __init__(self, args):
        super(AttGAN, self).__init__(args)
        dropout = 0.5 if not args.no_dropout else 0.0
        gen_norm = args.gen_norm if args.gen_norm is not None else 'instance'
        self.model.gen = networks.MultiDomainResnetGenerator(args.input_nc, args.output_nc, norm_layer=gen_norm,
                                        dropout=dropout, num_downs=2, n_blocks=9, num_domains=args.num_domains, sn=args.gen_sn)
        self.model.att = networks.UpsampleAttn(args.input_nc, norm_layer=gen_norm, num_downs=2, n_blocks=3)
        if 'train' in args.mode:
            self.model.dis = networks.NoNormDiscriminator(args.input_nc, num_domains=args.num_domains, image_size=args.crop_size, sn=args.dis_sn)
            # create optimizers
            for net in self.model:
                self.optimizer[net] = torch.optim.Adam(self.model[net].parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
            # define classification loss
            self.gan_loss = loss.GANLoss(args.gan_mode)
            self.cls_loss = nn.BCEWithLogitsLoss()
            self.l1_loss = nn.L1Loss()
            self.preceptual_loss = loss.VGGPerceptualLoss(args.crop_size).to(self.device)

    def set_inputs(self, inputs):
        self.real_a = inputs['x1'].to(self.device).detach()
        self.cls_a = inputs['y1'].to(self.device).detach()
        self.real_b = inputs['x2'].to(self.device).detach()
        self.cls_b = inputs['y2'].to(self.device).detach()
        self.real = torch.cat((self.real_a, self.real_b), dim=0)
        self.c_org = torch.cat((self.cls_a, self.cls_b), dim=0)

    def forward_generate(self, x, c):
        att = self.model.att(x)
        fg = att * x
        bg = (1-att) * x
        fake = self.model.gen(fg, c)
        return fake * att + bg, fake, att

    def forward(self):
        # A -> B' -> A''
        self.fake_b_att, self.fake_b, self.att_a = self.forward_generate(self.real_a, self.cls_b)
        self.rec_a_att, self.rec_a, self.att_rec_b = self.forward_generate(self.fake_b, self.cls_a)
        # B -> A' -> B''
        self.fake_a_att, self.fake_a, self.att_b = self.forward_generate(self.real_b, self.cls_a)
        self.rec_b_att, self.rec_b, self.att_rec_a = self.forward_generate(self.fake_a, self.cls_b)

    def update_D(self):
        # with fake images
        self.optimizer.dis.zero_grad()
        fake = torch.cat((self.fake_a, self.fake_b), dim=0)
        loss_d_fake = self.backward_D(self.model.dis, self.real, fake, self.c_org)
        fake = torch.cat((self.rec_a, self.rec_b), dim=0)
        loss_d_rec = self.backward_D(self.model.dis, self.real, fake, self.c_org)
        loss_d = loss_d_fake + loss_d_rec
        loss_d.backward()
        self.optimizer.dis.step()
        self.loss.loss_d = loss_d.item()

    def backward_D(self, netD, real, fake, c_org):
        # with fake images
        pred_fake, _ = netD(fake.detach())
        loss_fake = self.gan_loss(pred_fake, 0)
        # with real images
        pred_real, pred_real_cls = netD(real)
        loss_real = self.gan_loss(pred_real, 1)
        # adv loss
        loss_d_adv = loss_fake + loss_real
        # classification loss
        loss_d_cls = self.cls_loss(pred_real_cls, c_org)
        # total dis loss
        loss_d = loss_d_adv + self.args.lambda_cls * loss_d_cls
        return loss_d

    def update_G(self):
        self.optimizer.gen.zero_grad()
        self.optimizer.att.zero_grad()
        self.backward_G()
        self.optimizer.gen.step()
        self.optimizer.att.step()

    def backward_G(self):
        fake = torch.cat((self.fake_a, self.fake_b), dim=0)
        recon = torch.cat((self.rec_a, self.rec_b), dim=0)
        # adv for generator
        pred_fake, pred_fake_cls = self.model.dis(fake)
        loss_g_adv = self.gan_loss(pred_fake, 1)
        # adv for rec
        pred_fake, pred_fake_cls = self.model.dis(recon)
        loss_g_adv_rec = self.gan_loss(pred_fake, 1)
        # classification
        loss_g_cls = self.cls_loss(pred_fake_cls, self.c_org) * self.args.lambda_cls_g
        # reconstruction loss
        loss_cc = self.l1_loss(self.real, recon) * self.args.lambda_rec
        # perceptual loss
        loss_per = self.preceptual_loss(self.real, fake) * self.args.lambda_perceptual
        # total G loss
        loss_g = loss_g_adv + loss_g_cls + loss_cc + loss_per + loss_g_adv_rec
        loss_g.backward()
        self.loss.g_adv = loss_g_adv.item()
        self.loss.g_cls = loss_g_cls.item()
        self.loss.g_cc = loss_cc.item()
        self.loss.perceptual = loss_per.item()

    def compute_visuals(self):
        images_a = self.normalize_image(self.real_a).detach()
        images_b = self.normalize_image(self.real_b).detach()
        images_a1 = self.normalize_image(self.att_a).detach()
        images_a1 = images_a1.repeat((1,3,1,1))
        images_a2 = self.normalize_image(self.fake_a).detach()
        images_a3 = self.normalize_image(self.rec_a).detach()
        images_b1 = self.normalize_image(self.att_b).detach()
        images_b1 = images_b1.repeat((1,3,1,1))
        images_b2 = self.normalize_image(self.fake_b).detach()
        images_b3 = self.normalize_image(self.rec_b).detach()
        row1 = torch.cat((images_a[0:1, ::], images_a1[0:1, ::], images_b2[0:1, ::], images_a3[0:1, ::]), dim=3)
        row2 = torch.cat((images_b[0:1, ::], images_b1[0:1, ::], images_a2[0:1, ::], images_b3[0:1, ::]), dim=3)
        return torch.cat((row1,row2), dim=2)

    def optimize_parameters(self, global_iter):
        self.forward()
        if global_iter % self.args.d_iter != 0:
            self.update_D()
        else:
            self.update_D()
            self.update_G()

    def normalize_image(self, x):
        return x[:,0:3,:,:]