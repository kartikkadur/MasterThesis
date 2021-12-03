import os
import networks
import torch
import torch.nn as nn
import numpy as np
import loss
from models.model import Model

class DRIT(Model):
    def __init__(self, args):
        super(DRIT, self).__init__(args)
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
            self.model.att_enc = networks.AttributeEncoderConcat(args.input_nc, output_nc=self.latent_dim, num_domains=args.num_domains, \
                norm_layer=None, activation='leaky_relu')
            self.model.gen = networks.GeneratorConcat(args.input_nc, num_domains=args.num_domains, latent_dim=self.latent_dim)
        else:
            self.model.att_enc = networks.AttributeEncoder(args.input_nc, output_nc=self.latent_dim, num_domains=args.num_domains)
            self.model.gen = networks.Generator(args.input_nc, latent_dim=self.latent_dim, num_domains=args.num_domains)
        # init discriminator only when training
        if 'train' in args.mode:
            self.model.dis1 = networks.Discriminator(args.input_nc, norm_layer=args.dis_norm, sn=args.dis_spectral_norm, num_domains=args.num_domains, image_size=args.crop_size)
            self.model.dis2 = networks.Discriminator(args.input_nc, norm_layer=args.dis_norm, sn=args.dis_spectral_norm, num_domains=args.num_domains, image_size=args.crop_size)
            # create optimizers
            for net in self.model:
                self.optimizer[net] = torch.optim.Adam(self.model[net].parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.0001)
            # use content discriminator
            if self.args.use_dis_content:
                self.model.cont_dis = networks.ContentDiscriminator(num_domains=args.num_domains) 
                self.optimizer.cont_dis = torch.optim.Adam(self.model.cont_dis.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
            # define losses
            self.gan_loss = loss.GANLoss(args.gan_mode)
            self.cls_loss = nn.BCEWithLogitsLoss()
            self.l1_loss = nn.L1Loss()

    def adversarial_loss(self, inp, trg):
        assert trg in [0,1]
        out = torch.nn.functional.sigmoid(inp)
        if trg == 0:
            label = torch.zeros_like(out).to(self.device)
        else:
            label = torch.ones_like(out).to(self.device)
        adv_loss = torch.nn.functional.binary_cross_entropy(out, label)
        return adv_loss

    def get_z_random(self, bs, latent_dim, random_type='gauss'):
        z = torch.randn(bs, latent_dim).to(self.device)
        return z

    def set_inputs(self, inputs):
        self.real_a = inputs['x1'].to(self.device).detach()
        self.cls_a = inputs['y1'].to(self.device).detach()
        self.real_b = inputs['x2'].to(self.device).detach()
        self.cls_b = inputs['y2'].to(self.device).detach()

    def forward_content(self, img1, img2):
        z_content_a = self.model.cont_enc(img1)
        z_content_b = self.model.cont_enc(img2)
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

    def forward_random(self, img, c_org):
        z_content = self.model.cont_enc(img)
        outputs = []
        for i in range(self.args.num_domains):
            z_random = self.get_z_random(image.size(0), self.latent_dim)
            c_trg = np.zeros((image.size(0), self.args.num_domains))
            c_trg[:,i] = 1
            c_trg = torch.FloatTensor(c_trg).to(self.device)
            output = self.model.gen(z_content, z_random, c_trg)
            outputs.append(output)
        return outputs

    def forward_reference(self, img, ref, c_trg):
        z_content = self.model.cont_enc(img)
        mu, logvar = self.model.att_enc(ref, c_trg)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z_attr = eps.mul(std).add_(mu)
        output = self.model.gen(z_content, z_attr, c_trg)
        return output

    def forward(self):
        # content encoder outputs
        self.z_content_a, self.z_content_b = self.forward_content(self.real_a, self.real_b)
        # attribute encoder outputs
        if self.concat:
            self.z_attr_a, self.mu_a, self.logvar_a = self.forward_attribute(self.real_a, self.cls_a)
            self.z_attr_b, self.mu_b, self.logvar_b = self.forward_attribute(self.real_b, self.cls_b)
        else:
            self.z_attr_a = self.forward_attribute(self.real_a, self.cls_a)
            self.z_attr_b = self.forward_attribute(self.real_b, self.cls_b)
        # get random z_a
        self.z_random = self.get_z_random(self.args.batch_size, self.latent_dim)
        # first cross translation
        # for domain a
        self.fake_a_encoded = self.forward_gen(self.z_content_b, self.z_attr_a, self.cls_a)
        self.fake_aa_encoded = self.forward_gen(self.z_content_a, self.z_attr_a, self.cls_a)
        self.fake_a_random = self.forward_gen(self.z_content_b, self.z_random, self.cls_a)
        # for domain b
        self.fake_b_encoded = self.forward_gen(self.z_content_a, self.z_attr_b, self.cls_b)
        self.fake_bb_encoded = self.forward_gen(self.z_content_b, self.z_attr_b, self.cls_b)
        self.fake_b_random = self.forward_gen(self.z_content_a, self.z_random, self.cls_b)
        # get reconstructed encoded z_c
        self.z_content_recon_b, self.z_content_recon_a = self.forward_content(self.fake_a_encoded, self.fake_b_encoded)
        # get reconstructed encoded z_a
        if self.concat:
            self.z_attr_recon_a, self.mu_recon_a, self.logvar_recon_a = self.forward_attribute(self.fake_a_encoded, self.cls_a)
            self.z_attr_recon_b, self.mu_recon_b, self.logvar_recon_b = self.forward_attribute(self.fake_b_encoded, self.cls_b)
        else:
            self.z_attr_recon_a = self.forward_attribute(self.fake_a_encoded, self.cls_a)
            self.z_attr_recon_b = self.forward_attribute(self.fake_b_encoded, self.cls_b)
        # second cross translation
        self.fake_a_recon = self.model.gen(self.z_content_recon_a, self.z_attr_recon_a, self.cls_a)
        self.fake_b_recon = self.model.gen(self.z_content_recon_b, self.z_attr_recon_b, self.cls_b)
        # for latent regression
        if 'train' in self.args.mode:
            if self.concat:
                self.mu2_a, _ = self.model.att_enc(self.fake_a_random, self.cls_a)
                self.mu2_b, _ = self.model.att_enc(self.fake_b_random, self.cls_b)
            else:
                self.z_attr_random_a = self.model.att_enc(self.fake_a_random, self.cls_a)
                self.z_attr_random_b = self.model.att_enc(self.fake_b_random, self.cls_b)
 
    def update_D_content(self):
        self.z_content_a = self.model.cont_enc(self.real_a)
        self.z_content_b = self.model.cont_enc(self.real_b)
        self.optimizer.cont_dis.zero_grad()
        pred_cls_a = self.model.cont_dis(self.z_content_a.detach())
        pred_cls_b = self.model.cont_dis(self.z_content_b.detach())
        loss_d_content_a = self.cls_loss(pred_cls_a, self.cls_a)
        loss_d_content_b = self.cls_loss(pred_cls_b, self.cls_b)
        loss_d_content = loss_d_content_a + loss_d_content_b
        loss_d_content.backward()
        self.loss_d_content = loss_d_content.item()
        nn.utils.clip_grad_norm_(self.model.cont_dis.parameters(), 5)
        self.optimizer.cont_dis.step()

    def update_D(self):
        self.forward()
        # with encoded images
        self.optimizer.dis1.zero_grad()
        self.d1_gan_loss, self.d1_cls_loss = self.backward_D(self.model.dis1, self.real_a, self.real_b, self.fake_a_encoded, self.fake_b_encoded)
        self.optimizer.dis1.step()
        # with latent generated images
        self.optimizer.dis2.zero_grad()
        self.d2_gan_loss, self.d2_cls_loss = self.backward_D(self.model.dis2, self.real_a, self.real_b, self.fake_a_random, self.fake_b_random)
        self.optimizer.dis2.step()

    def backward_D(self, netD, real_a, real_b, fake_a, fake_b):
        # with fake images
        pred_fake_a, pred_fake_cls_a = netD.forward(fake_a.detach())
        pred_fake_b, pred_fake_cls_b = netD.forward(fake_b.detach())
        loss_fake_a = self.gan_loss(pred_fake_a, 0)
        loss_fake_b = self.gan_loss(pred_fake_b, 0)
        loss_fake = loss_fake_a + loss_fake_b
        # with real images
        pred_real_a, pred_real_cls_a = netD.forward(real_a)
        pred_real_b, pred_real_cls_b = netD.forward(real_b)
        loss_real_a = self.gan_loss(pred_real_a, 1)
        loss_real_b = self.gan_loss(pred_real_b, 1)
        loss_real = loss_real_a + loss_real_b
        # adv loss
        loss_d_adv = loss_fake + loss_real
        # classification loss
        loss_cls_a = self.cls_loss(pred_real_cls_a, self.cls_a)
        loss_cls_b = self.cls_loss(pred_real_cls_b, self.cls_b)
        loss_d_cls = loss_cls_a + loss_cls_b
        # dis loss
        loss_d = loss_d_adv + self.args.lambda_cls * loss_d_cls 
        loss_d.backward()
        return loss_d_adv, loss_d_cls

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
        if self.args.use_dis_content:
            loss_g_content = self.backward_G_GAN_content(self.z_content_a, self.z_content_b)
        # Ladv for generator
        pred_fake_a, pred_fake_cls_a = self.model.dis1(self.fake_a_encoded)
        pred_fake_b, pred_fake_cls_b = self.model.dis1(self.fake_b_encoded)
        loss_g_adv_a = self.gan_loss(pred_fake_a, 1)
        loss_g_adv_b = self.gan_loss(pred_fake_b, 1)
        loss_g_adv = loss_g_adv_a + loss_g_adv_b    
        # classification
        loss_g_cls_a = self.cls_loss(pred_fake_cls_a, self.cls_a)
        loss_g_cls_b = self.cls_loss(pred_fake_cls_b, self.cls_b)
        loss_g_cls = (loss_g_cls_a + loss_g_cls_b) * self.args.lambda_cls_g
        # self and cross-cycle recon
        loss_g_self_a = self.l1_loss(self.real_a, self.fake_aa_encoded)
        loss_g_self_b = self.l1_loss(self.real_b, self.fake_bb_encoded)
        loss_g_self = (loss_g_self_a + loss_g_self_b) * self.args.lambda_rec
        loss_g_cc_a = self.l1_loss(self.real_a, self.fake_a_recon)
        loss_g_cc_b = self.l1_loss(self.real_b, self.fake_b_recon)
        loss_g_cc = (loss_g_cc_a + loss_g_cc_b) * self.args.lambda_rec
        # KL loss - z_c
        loss_kl_zc_a = self._l2_regularize(self.z_content_a)
        loss_kl_zc_b = self._l2_regularize(self.z_content_b)
        loss_kl_zc = (loss_kl_zc_a + loss_kl_zc_b) * 0.01
        # KL loss - z_a
        if self.concat:
            mu = torch.cat((self.mu_a, self.mu_b), dim=0)
            logvar = torch.cat((self.logvar_a, self.logvar_b), dim=0)
            kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
        else:
            z_attr = torch.cat((self.z_attr_a, self.z_attr_b), dim=0)
            loss_kl_za = self._l2_regularize(z_attr) * 0.01
        # total loss G
        loss_g = loss_g_adv + loss_g_cls + loss_g_self + loss_g_cc + loss_kl_zc + loss_kl_za
        if self.args.use_dis_content:
            loss_g += loss_g_content
        loss_g.backward(retain_graph=True)

        self.loss_g_adv = loss_g_adv.item()
        self.loss_g_cls = loss_g_cls.item()
        if self.args.use_dis_content:
            self.loss_g_content = loss_g_content.item()
        self.kl_loss_zc = loss_kl_zc.item()
        self.kl_loss_za = loss_kl_za.item()
        self.l1_self_rec_loss = loss_g_self.item()
        self.l1_cc_rec_loss = loss_g_cc.item()
        self.loss_g = loss_g.item()

    def backward_G_GAN_content(self, cont_a, cont_b):
        pred_cls_a = self.model.cont_dis(cont_a)
        pred_cls_b = self.model.cont_dis(cont_b)
        loss_g_content_a = self.cls_loss(pred_cls_a, 1-self.cls_a)
        loss_g_content_b = self.cls_loss(pred_cls_b, 1-self.cls_b)
        loss_g_content = loss_g_content_a + loss_g_content_b
        return loss_g_content

    def backward_G_alone(self):
        # Ladv for generator
        pred_fake_a, pred_fake_cls_a = self.model.dis2(self.fake_a_random)
        pred_fake_b, pred_fake_cls_b = self.model.dis2(self.fake_b_random)
        loss_g_adv_a = self.gan_loss(pred_fake_a, 1)
        loss_g_adv_b = self.gan_loss(pred_fake_b, 1)
        loss_g_adv2 = loss_g_adv_a + loss_g_adv_b
        # classification
        loss_g_cls_a = self.cls_loss(pred_fake_cls_a, self.cls_a)
        loss_g_cls_b = self.cls_loss(pred_fake_cls_b, self.cls_b)
        loss_g_cls2 = (loss_g_cls_a + loss_g_cls_b) * self.args.lambda_cls_g
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
        loss_g.backward()
        self.l1_recon_z_loss = loss_z_l1.item()
        self.gan2_loss = loss_g_adv2.item()
        self.gan2_cls_loss = loss_g_cls2.item()

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

    def print_losses(self, block):
        block.log(f"G adv Loss : {self.loss_g_adv}, G cls loss: {self.loss_g_cls}, G cycle consistency : {self.l1_cc_rec_loss}")
        block.log('\n')