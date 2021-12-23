import os

from numpy.core.numeric import normalize_axis_tuple
import networks
import torch
import torch.nn as nn
import numpy as np
import loss
from models.model import Model

class BaseModel(Model):
    def __init__(self, args):
        super(BaseModel, self).__init__(args)
        self.latent_dim = args.latent_dim
        # lr for content discriminator
        lr_dcontent = args.lr/2.5
        if args.concat:
            self.concat = True
        else:
            self.concat = False
        # create models
        self.model.cont_enc = networks.ContentEncoder(args.input_nc)
        if self.concat:
            self.model.att_enc = networks.AttributeEncoderConcat(args.input_nc, output_nc=self.latent_dim,
                                                num_domains=args.num_domains, norm_layer=None, activation='leaky_relu')
            self.model.gen = networks.GeneratorConcat(args.input_nc, num_domains=args.num_domains,
                                                    latent_dim=self.latent_dim, upsample_layer=args.dec_upsample)
        else:
            self.model.att_enc = networks.AttributeEncoder(args.input_nc, output_nc=self.latent_dim, num_domains=args.num_domains)
            self.model.gen = networks.Generator(args.input_nc, latent_dim=self.latent_dim, num_domains=args.num_domains)
        # create discriminators, optimizers and loss only while training
        if 'train' in args.mode:
            if args.ms_dis:
                self.model.dis1 = networks.MultiScaleDiscriminator(args.input_nc, norm_layer=args.dis_norm, sn=args.dis_sn,
                                                                num_domains=args.num_domains, num_scales=args.num_scales)
                self.model.dis2 = networks.MultiScaleDiscriminator(args.input_nc, norm_layer=args.dis_norm, sn=args.dis_sn,
                                                                num_domains=args.num_domains, num_scales=args.num_scales)
            else:
                self.model.dis1 = networks.Discriminator(args.input_nc, norm_layer=args.dis_norm,
                                        sn=args.dis_sn, num_domains=args.num_domains, image_size=args.crop_size)
                self.model.dis2 = networks.Discriminator(args.input_nc, norm_layer=args.dis_norm,
                                        sn=args.dis_sn, num_domains=args.num_domains, image_size=args.crop_size)
            # create optimizers
            for net in self.model:
                self.optimizer[net] = torch.optim.Adam(self.model[net].parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0001)
            # use content discriminator
            if self.args.use_dis_content:
                self.model.cont_dis = networks.ContentDiscriminator(num_domains=args.num_domains) 
                self.optimizer.cont_dis = torch.optim.Adam(self.model.cont_dis.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
            # define losses
            self.gan_loss = loss.GANLoss(args.gan_mode).to(self.device)
            self.cls_loss = nn.BCEWithLogitsLoss().to(self.device)
            self.l1_loss = nn.L1Loss().to(self.device)
            if args.vgg_loss is not None:
                self.perceptual_loss = loss.VGGPerceptualLoss(args).to(self.device)
        self.print_loss = ['g_adv', 'g_cls', 'l1_cc_rec']

    def get_z_random(self, bs, latent_dim):
        z = torch.randn(bs, latent_dim).to(self.device)
        return z

    def set_inputs(self, inputs):
        self.real_a = inputs['x1'].to(self.device).detach()
        self.cls_a = inputs['y1'].to(self.device).detach()
        self.real_b = inputs['x2'].to(self.device).detach()
        self.cls_b = inputs['y2'].to(self.device).detach()
        # combined image
        self.real = torch.cat((self.real_a, self.real_b), dim=0)
        self.c_org = torch.cat((self.cls_a, self.cls_b), dim=0)

    def forward_content(self, img1, img2):
        img = torch.cat((img1, img2), dim=0)
        z_content = self.model.cont_enc(img)
        z_content_a, z_content_b = torch.split(z_content, self.args.batch_size, dim=0)
        return z_content_a, z_content_b

    def forward_attribute(self, real, c_org):
        if self.concat:
            mu, logvar = self.model.att_enc(real, c_org)
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1))
            z_attr = eps.mul(std).add_(mu)
            return z_attr, mu, logvar
        else:
            z_attr = self.model.att_enc(real, c_org)
            return z_attr

    def forward_gen(self, cont, att, c_org):
        fake = self.model.gen(cont, att, c_org)
        return fake

    def forward(self):
        # content encoder outputs
        self.z_content_a, self.z_content_b = self.forward_content(self.real_a, self.real_b)
        # attribute encoder outputs
        if self.concat:
            self.z_attr, self.mu, self.logvar = self.forward_attribute(self.real, self.c_org)
            self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, self.args.batch_size, dim=0)
        else:
            self.z_attr = self.forward_attribute(self.real, self.c_org)
            self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, self.args.batch_size, dim=0)
        # get random z_a
        self.z_random = self.get_z_random(self.args.batch_size, self.latent_dim)
        # first cross translation
        # for domain a
        cont = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b), dim=0)
        attr = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random), dim=0)
        c = torch.cat((self.cls_a, self.cls_a, self.cls_a), dim=0)
        fake = self.forward_gen(cont, attr, c)
        self.fake_a_encoded, self.fake_aa_encoded, self.fake_a_random = torch.split(fake, self.args.batch_size, dim=0)

        # for domain b
        cont = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a), dim=0)
        attr = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random), dim=0)
        c = torch.cat((self.cls_b, self.cls_b, self.cls_b), dim=0)
        fake = self.forward_gen(cont, attr, c)
        self.fake_b_encoded, self.fake_bb_encoded, self.fake_b_random = torch.split(fake, self.args.batch_size, dim=0)
        self.fake_encoded = torch.cat((self.fake_a_encoded, self.fake_b_encoded), dim=0)
        self.fake_random = torch.cat((self.fake_a_random, self.fake_b_random), dim=0)
        # get reconstructed encoded z_c
        self.z_content_recon_b, self.z_content_recon_a = self.forward_content(self.fake_a_encoded, self.fake_b_encoded)
        # get reconstructed encoded z_a
        if self.concat:
            z_attr_recon, mu_recon, logvar_recon = self.forward_attribute(self.fake_encoded, self.c_org)
        else:
            z_attr_recon = self.forward_attribute(self.fake_encoded, self.c_org)
        # second cross translation
        z_content_recon = torch.cat((self.z_content_recon_a, self.z_content_recon_b), dim=0)
        self.fake_recon = self.model.gen(z_content_recon, z_attr_recon, self.c_org)
        self.fake_a_recon, self.fake_b_recon = torch.split(self.fake_recon, self.args.batch_size, dim=0)
        
        # for latent regression
        if 'train' in self.args.mode:
            if self.concat:
                self.mu2, _ = self.model.att_enc(self.fake_random, self.c_org)
                self.mu2_a, self.mu2_b = torch.split(self.mu2, self.args.batch_size, dim=0)
            else:
                z_attr_random = self.model.att_enc(self.fake_random, self.c_org)
                self.z_attr_random_a, self.z_attr_random_b = torch.split(z_attr_random, self.args.batch_size, dim=0)
 
    def update_D_content(self):
        self.z_content = self.model.cont_enc(self.real)
        self.z_content_a, self.z_content_b = torch.split(self.z_content, self.args.batch_size)
        self.optimizer.cont_dis.zero_grad()
        pred_cls = self.model.cont_dis(self.z_content.detach())
        loss_d_content = self.cls_loss(pred_cls, self.c_org)
        loss_d_content.backward()
        self.loss_d_content = loss_d_content.item()
        nn.utils.clip_grad_norm_(self.model.cont_dis.parameters(), 5)
        self.optimizer.cont_dis.step()

    def update_D(self):
        self.forward()
        if self.args.ms_dis:
            backward_fn = self.backward_D_multi_scale
        else:
            backward_fn = self.backward_D
        # with encoded images
        self.optimizer.dis1.zero_grad()
        self.d1_gan_loss, self.d1_cls_loss = backward_fn(self.model.dis1, self.real, self.fake_encoded, self.c_org)
        self.optimizer.dis1.step()
        # with latent generated images
        self.optimizer.dis2.zero_grad()
        self.d2_gan_loss, self.d2_cls_loss = backward_fn(self.model.dis2, self.real, self.fake_random, self.c_org)
        self.optimizer.dis2.step()

    def backward_D_multi_scale(self, netD, real, fake, c_org):
        # with fake images
        outputs_fake = netD(fake.detach())
        outputs_real = netD(real)
        loss_d_adv = 0
        loss_d_cls = 0
        for i, (out0, out1) in enumerate(zip(outputs_fake, outputs_real)):
            # gan loss
            loss_d_adv += self.gan_loss(out0[0], 0)
            loss_d_adv += self.gan_loss(out1[0], 1)
            # class loss
            loss_d_cls += self.cls_loss(out1[1], c_org)
        # dis loss
        loss_d = loss_d_adv + self.args.lambda_cls * loss_d_cls
        loss_d.backward()
        return loss_d_adv.item(), loss_d_cls.item()

    def backward_D(self, netD, real, fake, c_org):
        # with fake images
        pred_fake, pred_fake_cls = netD(fake.detach())
        loss_fake = self.gan_loss(pred_fake, 0)
        # with real images
        pred_real, pred_real_cls = netD(real)
        loss_real = self.gan_loss(pred_real, 1)
        # adv loss
        loss_d_adv = loss_fake + loss_real
        # classification loss
        loss_d_cls = self.cls_loss(pred_real_cls, c_org)
        # dis loss
        loss_d = loss_d_adv + self.args.lambda_cls * loss_d_cls
        loss_d.backward()
        return loss_d_adv.item(), loss_d_cls.item()

    def update_EG(self):
        # update G, Ec, Ea
        self.optimizer.cont_enc.zero_grad()
        self.optimizer.att_enc.zero_grad()
        self.optimizer.gen.zero_grad()
        self.backward_EG()
        self.optimizer.cont_enc.step()
        self.optimizer.att_enc.step()
        self.optimizer.gen.step()
        # call forward again
        self.forward()
        # update G, Ec
        self.optimizer.cont_enc.zero_grad()
        self.optimizer.gen.zero_grad()
        self.backward_G_alone()
        self.optimizer.cont_enc.step()
        self.optimizer.gen.step()

    def backward_EG(self):
        # content Ladv for generator
        z_content = torch.cat((self.z_content_a, self.z_content_b), dim=0)
        if self.args.use_dis_content:
            loss_g_content = self.backward_G_GAN_content(z_content, self.c_org)
        # Ladv for generator
        if self.args.ms_dis:
            outputs_fake = self.model.dis1(self.fake_encoded)
            loss_g_adv = 0
            loss_g_cls = 0
            for out0 in outputs_fake:
                loss_g_adv += self.gan_loss(out0[0], 1)
                loss_g_cls += self.cls_loss(out0[1], self.c_org)
            loss_g_cls *= self.args.lambda_cls_G
        else:
            pred_fake, pred_fake_cls = self.model.dis1(self.fake_encoded)
            loss_g_adv = self.gan_loss(pred_fake, 1)
            # classification
            loss_g_cls = self.cls_loss(pred_fake_cls, self.c_org) * self.args.lambda_cls_G
        # self recon
        fake = torch.cat((self.fake_aa_encoded, self.fake_bb_encoded), dim=0)
        loss_g_self = self.l1_loss(self.real, fake) * self.args.lambda_rec
        # cross-cycle recon
        loss_g_cc = self.l1_loss(self.real, self.fake_recon) * self.args.lambda_rec
        # perceptual loss
        if self.args.vgg_loss is not None:
            loss_g_p = self.perceptual_loss(self.real, self.fake_encoded) * self.args.lambda_perceptual
        # KL loss - z_c
        loss_kl_zc = self._l2_regularize(z_content) * 0.01
        # KL loss - z_a
        if self.concat:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
        else:
            z_attr = torch.cat((self.z_attr_a, self.z_attr_b), dim=0)
            loss_kl_za = self._l2_regularize(z_attr) * 0.01
        # total loss G
        loss_g = loss_g_adv + loss_g_cls + loss_g_self + loss_g_cc + loss_kl_zc + loss_kl_za
        if self.args.use_dis_content:
            loss_g += loss_g_content
        if self.args.vgg_loss is not None:
            loss_g += loss_g_p
        loss_g.backward(retain_graph=True)

        self.loss.g_adv = loss_g_adv.item()
        self.loss.g_cls = loss_g_cls.item()
        if self.args.use_dis_content:
            self.loss.g_content = loss_g_content.item()
        self.loss.kl_zc = loss_kl_zc.item()
        self.loss.kl_za = loss_kl_za.item()
        self.loss.l1_self_rec = loss_g_self.item()
        self.loss.l1_cc_rec = loss_g_cc.item()
        self.loss.total_g = loss_g.item()

    def backward_G_GAN_content(self, cont, c_org):
        pred_cls = self.model.cont_dis(cont)
        loss_g_content = self.cls_loss(pred_cls, 1 - c_org)
        return loss_g_content

    def backward_G_alone(self):
        # Ladv for generator
        fake = torch.cat((self.fake_a_random, self.fake_b_random), dim=0)
        if self.args.ms_dis:
            outputs_fake = self.model.dis1(fake)
            loss_g_adv2 = 0
            loss_g_cls2 = 0
            for out0 in outputs_fake:
                loss_g_adv2 += self.gan_loss(out0[0], 1)
                loss_g_cls2 += self.cls_loss(out0[1], self.c_org)
            loss_g_cls2 *= self.args.lambda_cls_G
        else:
            pred_fake, pred_fake_cls = self.model.dis2(fake)
            loss_g_adv2 = self.gan_loss(pred_fake, 1)
            # classification
            loss_g_cls2 = self.cls_loss(pred_fake_cls, self.c_org) * self.args.lambda_cls_G
        # perceptual
        if self.args.vgg_loss is not None:
            loss_g_p =  self.perceptual_loss(self.real, self.fake_random) * self.args.lambda_perceptual
        # latent regression loss
        if self.concat:
            loss_z_l1_a = self.l1_loss(self.mu2_a, self.z_random)
            loss_z_l1_b = self.l1_loss(self.mu2_b, self.z_random)
        else:
            loss_z_l1_a = self.l1_loss(self.z_attr_random_a, self.z_random)
            loss_z_l1_b = self.l1_loss(self.z_attr_random_b, self.z_random)
        loss_z_l1 = (loss_z_l1_a + loss_z_l1_b) * 10
        # total G loss
        loss_g = loss_z_l1 + loss_g_adv2 + loss_g_cls2
        if self.args.vgg_loss is not None:
            loss_g += loss_g_p
        loss_g.backward()
        self.loss.l1_recon_z = loss_z_l1.item()
        self.loss.gan2 = loss_g_adv2.item()
        self.loss.gan2_cls = loss_g_cls2.item()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def compute_visuals(self):
        images_a = self.normalize_image(self.real_a).detach()
        images_b = self.normalize_image(self.real_b).detach()
        images_a1 = self.normalize_image(self.fake_a_encoded).detach()
        images_a2 = self.normalize_image(self.fake_a_random).detach()
        images_a3 = self.normalize_image(self.fake_a_recon).detach()
        images_a4 = self.normalize_image(self.fake_aa_encoded).detach()
        images_b1 = self.normalize_image(self.fake_b_encoded).detach()
        images_b2 = self.normalize_image(self.fake_b_random).detach()
        images_b3 = self.normalize_image(self.fake_b_recon).detach()
        images_b4 = self.normalize_image(self.fake_bb_encoded).detach()
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]), dim=3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]), dim=3)
        return torch.cat((row1,row2), dim=2)

    def normalize_image(self, x):
        return x[:,0:3,:,:]

    def optimize_parameters(self, global_iter):
        if self.args.use_dis_content:
            if global_iter % self.args.d_iter != 0:
                self.update_D_content()
            else:
                self.update_D()
                self.update_EG()
        else:
            self.update_D()
            self.update_EG()