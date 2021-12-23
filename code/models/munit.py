import os
import networks
import torch
import torch.nn as nn
import numpy as np
import loss
from models.model import Model

class MUNIT(Model):
    def __init__(self, args):
        super(MUNIT, self).__init__(args)
        self.model.style_enc = networks.MunitStyleEncoder(args.input_nc, args.ngf, 4, norm_layer=None,
                                        activation='relu', padding_type='reflect', num_domains=args.num_domains)
        self.model.cont_enc = networks.MunitContentEncoder(args.input_nc, args.ngf, 2, 4, norm_layer='instance',
                                        activation='relu', padding_type='reflect')
        self.model.decoder = networks.MunitDecoder(args.output_nc, self.model.cont_enc.output_nc, 2, 4,
                                norm_layer='adain', activation='relu', padding_type='reflect', num_domains=args.num_domains)
        if 'train' in args.mode:
            self.model.dis1 = networks.MutliScaleDiscriminator(args.input_nc, args.ndf, n_layer=4,
                                    activation='leaky_relu', padding_type='reflect', num_scales=3, num_domains=args.num_domains)
            self.model.dis2 = networks.MutliScaleDiscriminator(args.input_nc, args.ndf, n_layer=4,
                                    activation='leaky_relu', padding_type='reflect', num_scales=3, num_domains=args.num_domains)
            # create optimizers
            for net in self.model:
                self.optimizer[net] = torch.optim.Adam(self.model[net].parameters(), lr=args.lr, betas=(0, 0.9), weight_decay=0.0001)
            self.gan_loss = loss.GANLoss(args.gan_mode).to(self.device)
            self.cls_loss = nn.BCEWithLogitsLoss().to(self.device)
            self.l1_loss = nn.L1Loss().to(self.device)
            if args.vgg_loss is not None:
                self.perceptual_loss = loss.VGGPerceptualLoss(args).to(self.device)

    def set_inputs(self, inputs):
        self.real_a = inputs['x1'].to(self.device).detach()
        self.cls_a = inputs['y1'].to(self.device).detach()
        self.real_b = inputs['x2'].to(self.device).detach()
        self.cls_b = inputs['y2'].to(self.device).detach()
        self.real = torch.cat((self.real_a, self.real_b), dim=0)
        self.c_org = torch.cat((self.cls_a, self.cls_b), dim=0)

    def get_z_random(self, bs, latent_dim):
        z = torch.randn(bs, latent_dim).to(self.device)
        return z

    def forward_reference(self, src, ref, c):
        content = self.model.cont_enc(src)
        style = self.model.style_enc(ref, c)
        image = self.model.decoder(content, style, c)
        return image, content, style

    def forward_random(self, src, z_random, c):
        content = self.model.cont_enc(src)
        image = self.model.decoder(content, z_random, c)
        return image, content

    def forward(self):
        # content codes
        self.content = self.model.cont_enc(self.real)
        self.content_a, self.content_b = torch.split(self.content, self.args.batch_size, dim=0)
        # style codes
        self.style = self.model.style_enc(self.real, self.c_org)
        self.style_a, self.style_b = torch.split(self.style, self.args.batch_size, dim=0)
        # random style code
        self.z_random = self.get_z_random(self.args.batch_size, self.args.latent_dim)
        # first cross translation
        # b -> a', a -> a', b -> a' (random style)
        cont = torch.cat((self.content_b, self.content_a, self.content_b), dim=0)
        style = torch.cat((self.style_a, self.style_a, self.z_random), dim=0)
        c_org = torch.cat((self.cls_a, self.cls_a, self.cls_a), dim=0)
        fake_img = self.model.decoder(cont, style, c_org)
        self.fake_a_encoded, self.fake_aa_encoded, self.fake_a_random = torch.split(fake_img, self.args.batch_size, dim=0)
        # a -> b', b -> b', a -> b' (random style)
        cont = torch.cat((self.content_a, self.content_b, self.content_a), dim=0)
        style = torch.cat((self.style_b, self.style_b, self.z_random), dim=0)
        c_org = torch.cat((self.cls_b, self.cls_b, self.cls_b), dim=0)
        fake_img = self.model.decoder(cont, style, c_org)
        self.fake_b_encoded, self.fake_bb_encoded, self.fake_b_random = torch.split(fake_img, self.args.batch_size, dim=0)
        self.fake_encoded = torch.cat((self.fake_a_encoded, self.fake_b_encoded), dim=0)
        self.fake_random = torch.cat((self.fake_a_random, self.fake_b_random), dim=0)
        # reconstruct content and style
        fake_img = torch.cat((self.fake_b_encoded, self.fake_a_encoded), dim=0)
        self.cont_recon = self.model.cont_enc(fake_img)
        style_img = torch.cat((self.fake_a_encoded, self.fake_b_encoded), dim=0)
        self.style_recon = self.model.style_enc(style_img, self.c_org)
        self.fake_recon = self.model.decoder(self.cont_recon, self.style_recon, self.c_org)

    def update_D(self):
        self.forward()
        # with encoded images
        self.optimizer.dis1.zero_grad()
        self.backward_D(self.model.dis1, self.real, self.fake_encoded, self.c_org)
        self.optimizer.dis1.step()
        # with latent generated images
        self.optimizer.dis2.zero_grad()
        self.backward_D(self.model.dis2, self.real, self.fake_random, self.c_org)
        self.optimizer.dis2.step()

    def backward_D(self, netD, real, fake, c_org):
        # fake images
        pred_fake, pred_fake_cls = netD(fake.detach())
        # real images
        pred_real, pred_real_cls = netD(real)
        # losses
        loss_fake = 0
        loss_real = 0
        loss_cls = 0
        for p_fake, p_real, p_cls in zip(pred_fake, pred_real, pred_real_cls):
            loss_fake += self.gan_loss(p_fake, 0)
            loss_real += self.gan_loss(p_real, 0)
            loss_cls += self.cls_loss(p_cls, c_org)
        # adv loss
        loss_d_adv = loss_fake + loss_real
        # dis loss
        loss_d = loss_d_adv + self.args.lambda_cls * loss_cls
        loss_d.backward()
        self.loss.loss_d_adv = loss_d_adv.item()
        self.loss.loss_d_cls = loss_cls.item()

    def update_G(self):
        self.optimizer.cont_enc.zero_grad()
        self.optimizer.style_enc.zero_grad()
        self.optimizer.decoder.zero_grad()
        self.backward_reference()
        self.optimizer.cont_enc.step()
        self.optimizer.style_enc.step()
        self.optimizer.decoder.step()
        self.forward()
        self.optimizer.cont_enc.zero_grad()
        self.optimizer.decoder.zero_grad()
        self.backward_random()
        self.optimizer.cont_enc.step()
        self.optimizer.decoder.step()

    def backward_reference(self):
        # Ladv for generator
        pred_fake, pred_fake_cls = self.model.dis1(self.fake_encoded)
        loss_g_adv = 0
        loss_g_cls = 0
        for p_fake, p_cls in zip(pred_fake, pred_fake_cls):
            loss_g_adv += self.gan_loss(p_fake, 1)
            loss_g_cls += self.cls_loss(p_cls, self.c_org)
        # classification
        loss_g_cls = loss_g_cls * self.args.lambda_cls_G
        # self recon
        recon_self = torch.cat((self.fake_aa_encoded, self.fake_bb_encoded), dim=0)
        loss_g_self = self.l1_loss(self.real, recon_self) * self.args.lambda_rec
        # cross-cycle recon
        loss_g_cc = self.l1_loss(self.real, self.fake_recon) * self.args.lambda_rec
        # perceptual loss
        if self.args.vgg_loss is not None:
            loss_g_p = self.perceptual_loss(self.real, self.fake_encoded) * self.args.lambda_perceptual
        # KL loss - z_c
        content = torch.cat((self.content_a, self.content_b), dim=0)
        loss_kl_zc = self._l2_regularize(content) * 0.01
        # KL loss - z_a
        style = torch.cat((self.style_a, self.style_b), dim=0)
        loss_kl_za = self._l2_regularize(style) * 0.01
        # total loss G
        loss_g = loss_g_adv + loss_g_cls + loss_g_self + loss_g_cc + loss_kl_zc + loss_kl_za
        if self.args.vgg_loss is not None:
            loss_g += loss_g_p
        loss_g.backward(retain_graph=True)

        self.loss.g_adv = loss_g_adv.item()
        self.loss.g_cls = loss_g_cls.item()
        self.loss.kl_zc = loss_kl_zc.item()
        self.loss.kl_za = loss_kl_za.item()
        self.loss.l1_self_rec = loss_g_self.item()
        self.loss.l1_cc_rec = loss_g_cc.item()
        self.loss.total_g = loss_g.item()

    def backward_random(self):
        # Ladv for generator
        pred_fake, pred_fake_cls = self.model.dis2(self.fake_random)
        loss_g_adv2 = 0
        loss_g_cls2 = 0
        for p_fake, p_cls in zip(pred_fake, pred_fake_cls):
            loss_g_adv2 += self.gan_loss(p_fake, 1)
            loss_g_cls2 += self.cls_loss(p_cls, self.c_org)
        # perceptual
        if self.args.vgg_loss is not None:
            loss_g_p =  self.perceptual_loss(self.real, self.fake_random) * self.args.lambda_perceptual
        # total G loss
        loss_g = loss_g_adv2 + loss_g_cls2
        if self.args.vgg_loss is not None:
            loss_g += loss_g_p
        loss_g.backward()
        self.loss.gan2 = loss_g_adv2.item()
        self.loss.gan2_cls = loss_g_cls2.item()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def normalize_image(self, x):
        return x[:,0:3,:,:]

    def optimize_parameters(self, global_iter):
        self.update_D()
        self.update_G()