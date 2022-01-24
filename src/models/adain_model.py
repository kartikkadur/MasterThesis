import torch
import torch.nn as nn

from models.core import networks
from models.core import loss
from models.model import Model

class AdaINModel(Model):
    def __init__(self, args):
        super(AdaINModel, self).__init__(args)
        self.latent_dim = args.latent_dim
        self.style_dim = 64
        # create models
        self.model.content_encoder = networks.ContentEncoder(args.input_dim,
                                                             dim=args.dim)
        self.model.style_encoder = networks.StyleEncoderConcat(args.input_dim,
                                                                output_dim=self.style_dim,
                                                                dim=args.dim,
                                                                num_domains=args.num_domains,
                                                                norm_layer=None,
                                                                activation='lrelu')
        self.model.rand_style_encoder = networks.MappingNetwork(args.latent_dim,
                                                                self.style_dim,
                                                                args.num_domains)
        self.model.decoder = networks.AdaINDecoder(args.input_dim,
                                                    dim=self.model.content_encoder.output_dim, 
                                                    latent_dim=self.style_dim,
                                                    up_type=args.up_type,
                                                    dropout=args.use_dropout,
                                                    norm_layer=None)
        # create discriminators, optimizers and loss only while training
        if 'train' in args.mode:
            self.model.discriminator1 = networks.Discriminator(args.input_dim,
                                                                dim=args.dim,
                                                                norm_layer=args.dis_norm,
                                                                sn=args.dis_sn,
                                                                num_domains=args.num_domains,
                                                                image_size=args.crop_size)
            self.model.discriminator2 = networks.Discriminator(args.input_dim,
                                                                dim=args.dim,
                                                                norm_layer=args.dis_norm,
                                                                sn=args.dis_sn,
                                                                num_domains=args.num_domains,
                                                                image_size=args.crop_size)
        # create optimizers
            for net in self.model:
                self.optimizer[net] = torch.optim.Adam(self.model[net].parameters(),
                                                       lr=args.lr,
                                                       betas=(args.beta1, args.beta2),
                                                       weight_decay=args.wd)
            # use content discriminator
            if self.args.use_dis_content:
                # lr for content discriminator
                lr_dcontent = args.lr/2.5
                self.model.content_discriminator = networks.ContentDiscriminator(dim=self.model.content_encoder.output_dim,
                                                                                 num_domains=args.num_domains) 
                self.optimizer.content_discriminator = torch.optim.Adam(self.model.content_discriminator.parameters(),
                                                                        lr=lr_dcontent,
                                                                        betas=(args.beta1, args.beta2),
                                                                        weight_decay=args.wd)
            # define losses
            self.gan_loss = loss.GANLoss(args.gan_mode).to(self.device)
            self.classification_loss = nn.BCEWithLogitsLoss().to(self.device)
            self.l1_loss = nn.L1Loss().to(self.device)
            if args.vgg_loss is not None:
                self.perceptual_loss = loss.VGGPerceptualLoss(args.vgg_layers, args.layer_weights, args.vgg_type,
                                                                        args.vgg_loss, args.gpu_ids).to(self.device)
            self.print_loss = ['g_adv', 'g_cls', 'l1_cc_rec']
            if self.args.vgg_loss is not None:
                self.print_loss += ['g_p', 'g_p2']

    def get_z_random(self, bs, latent_dim):
        z = torch.randn(bs, latent_dim).to(self.device)
        return z

    def set_inputs(self, inputs):
        self.img_a = inputs['x1'].to(self.device).detach()
        self.cls_a = inputs['y1'].to(self.device).detach()
        self.img_b = inputs['x2'].to(self.device).detach()
        self.cls_b = inputs['y2'].to(self.device).detach()
        # combined image
        self.img = torch.cat((self.img_a, self.img_b), dim=0)
        self.c_org = torch.cat((self.cls_a, self.cls_b), dim=0)

    def forward_random(self, img, z_r, c_trg):
        trg = torch.zeros((self.args.num_domains,))
        trg[c_trg] = 1
        trg = trg.view(1, trg.size(0)).to(self.device)
        z_c = self.model.content_encoder(img)
        img_fake = self.model.decoder(z_c, z_r)
        return img_fake

    def forward_reference(self, img_src, img_ref, c_trg):
        trg = torch.zeros((self.args.num_domains,))
        trg[c_trg] = 1
        trg = trg.view(1, trg.size(0)).to(self.device)
        z_c = self.model.content_encoder(img_src)
        z_s, _, _ = self.model.style_encoder(img_ref, trg)
        img_fake = self.model.decoder(z_c, z_s)
        return img_fake

    def forward(self, img, c_org):
        z_c = self.model.content_encoder(img)
        z_ca, z_cb = torch.split(z_c, self.args.batch_size, dim=0)
        z_s, mu, logvar = self.model.style_encoder(img, c_org)
        z_sa, z_sb = torch.split(z_s, self.args.batch_size, dim=0)
        z_sr = self.get_z_random(c_org.size(0), self.args.latent_dim)
        z_r = self.model.rand_style_encoder(z_sr, c_org)
        z_ra, z_rb = torch.split(z_r, self.args.batch_size, dim=0)
        # translation from B -> A
        cls_a, cls_b = torch.split(c_org, self.args.batch_size, dim=0)
        content = torch.cat((z_cb, z_ca, z_cb), dim=0)
        style = torch.cat((z_sa, z_sa, z_ra), dim=0)
        trg_cls = torch.cat((cls_a, cls_a, cls_a), dim=0)
        fake_imgs = self.model.decoder(content, style)
        img_ba, img_aa, img_br = torch.split(fake_imgs, self.args.batch_size, dim=0)
        # translation from A -> B
        content = torch.cat((z_ca, z_cb, z_ca), dim=0)
        style = torch.cat((z_sb, z_sb, z_rb), dim=0)
        trg_cls = torch.cat((cls_b, cls_b, cls_b), dim=0)
        fake_imgs = self.model.decoder(content, style)
        img_ab, img_bb, img_ar = torch.split(fake_imgs, self.args.batch_size, dim=0)
        # concatinate images
        img_fake = torch.cat((img_ba, img_ab), dim=0)
        img_random = torch.cat((img_br, img_ar), dim=0)
        img_self = torch.cat((img_aa, img_bb), dim=0)
        return img_fake, img_random, img_self
 
    def update_content_discriminator(self, img, c_org):
        z_c = self.model.content_encoder(img)
        self.optimizer.content_discriminator.zero_grad()
        pred = self.model.content_discriminator(z_c.detach())
        loss_d_content = self.classification_loss(pred, c_org)
        loss_d_content.backward()
        self.loss_dc = loss_d_content.item()
        nn.utils.clip_grad_norm_(self.model.content_discriminator.parameters(), 5)
        self.optimizer.content_discriminator.step()

    def update_discriminator(self, img, c_org):
        backward_fn = self.backward_discriminator
        # construct images
        cls_a, cls_b = torch.split(c_org, self.args.batch_size, dim=0)
        z_c = self.model.content_encoder(img)
        z_ca, z_cb = torch.split(z_c, self.args.batch_size, dim=0)
        z_s, mu, logvar = self.model.style_encoder(img, c_org)
        z_sa, z_sb = torch.split(z_s, self.args.batch_size, dim=0)
        z_sr = self.get_z_random(c_org.size(0), self.args.latent_dim)
        z_r = self.model.rand_style_encoder(z_sr, c_org)
        z_ra, z_rb = torch.split(z_r, self.args.batch_size, dim=0)
        # translation from B -> A
        content = torch.cat((z_cb, z_cb), dim=0)
        style = torch.cat((z_sa, z_ra), dim=0)
        fake_imgs = self.model.decoder(content, style)
        img_ba, img_br = torch.split(fake_imgs, self.args.batch_size, dim=0)
        # translation from A -> B
        content = torch.cat((z_ca, z_ca), dim=0)
        style = torch.cat((z_sb, z_rb), dim=0)
        fake_imgs = self.model.decoder(content, style)
        img_ab, img_ar = torch.split(fake_imgs, self.args.batch_size, dim=0)
        # concat images
        img_fake = torch.cat((img_ba, img_ab), dim=0)
        img_random = torch.cat((img_br, img_ar), dim=0)
        # with fake images
        self.optimizer.discriminator1.zero_grad()
        backward_fn(self.model.discriminator1, img, img_fake, c_org)
        self.optimizer.discriminator1.step()
        # with latent generated images
        self.optimizer.discriminator2.zero_grad()
        backward_fn(self.model.discriminator2, img, img_random, c_org)
        self.optimizer.discriminator2.step()

    def backward_discriminator(self, netD, real, fake, c_org):
        # with fake images
        pred_fake, pred_fake_cls = netD(fake.detach())
        loss_fake = self.gan_loss(pred_fake, 0)
        # with real images
        pred_real, pred_real_cls = netD(real)
        loss_real = self.gan_loss(pred_real, 1)
        # adv loss
        loss_d_adv = loss_fake + loss_real
        # classification loss
        loss_d_cls = self.classification_loss(pred_real_cls, c_org)
        # dis loss
        loss_d = loss_d_adv + self.args.lambda_cls * loss_d_cls
        loss_d.backward()
        self.loss.d_adv = loss_d_adv.item()
        self.loss.d_cls = loss_d_cls.item()
        self.loss.d_total = loss_d.item()

    def update_generator(self, img, c_org):
        # update D, Ec, Ea
        self.optimizer.content_encoder.zero_grad()
        self.optimizer.style_encoder.zero_grad()
        self.optimizer.decoder.zero_grad()
        self.backward_generator(img, c_org)
        self.optimizer.content_encoder.step()
        self.optimizer.style_encoder.step()
        self.optimizer.decoder.step()
        # update D, Ec
        self.optimizer.content_encoder.zero_grad()
        self.optimizer.decoder.zero_grad()
        self.backward_decoder_random(img, c_org)
        self.optimizer.content_encoder.step()
        self.optimizer.decoder.step()

    def backward_generator(self, img, c_org):
        # encode content
        cls_a, cls_b = torch.split(c_org, self.args.batch_size, dim=0)
        z_c = self.model.content_encoder(img)
        z_ca, z_cb = torch.split(z_c, self.args.batch_size, dim=0)
        z_s, mu, logvar = self.model.style_encoder(img, c_org)
        z_sa, z_sb = torch.split(z_s, self.args.batch_size, dim=0)
        # translation from B -> A
        content = torch.cat((z_cb, z_ca), dim=0)
        style = torch.cat((z_sa, z_sa), dim=0)
        fake_imgs = self.model.decoder(content, style)
        img_ba, img_aa = torch.split(fake_imgs, self.args.batch_size, dim=0)
        # translation from A -> B
        content = torch.cat((z_ca, z_cb), dim=0)
        style = torch.cat((z_sb, z_sb), dim=0)
        fake_imgs = self.model.decoder(content, style)
        img_ab, img_bb = torch.split(fake_imgs, self.args.batch_size, dim=0)
        # concat images
        img_fake = torch.cat((img_ba, img_ab), dim=0)
        img_self = torch.cat((img_aa, img_bb), dim=0)
        # reconstruct images
        z_c_rec = self.model.content_encoder(img_fake)
        z_c_rec_b, z_c_rec_a = torch.split(z_c_rec, self.args.batch_size, dim=0)
        # style reconstruction
        z_s_rec, mu_rec, logvar_rec = self.model.style_encoder(img_fake, c_org)
        z_s_rec_a, z_s_rec_b = torch.split(z_s_rec, self.args.batch_size, dim=0)
        # image reconstruction
        content = torch.cat((z_c_rec_a, z_c_rec_b), dim=0)
        style = torch.cat((z_s_rec_a, z_s_rec_b), dim=0)
        img_recon = self.model.decoder(content, style)
        # content Ladv for generator
        if self.args.use_dis_content:
            loss_g_content = self.backward_content_discriminator(z_c)
        # Ladv for generator
        pred_fake, pred_fake_cls = self.model.discriminator1(img_fake)
        loss_g_adv = self.gan_loss(pred_fake, 1)
        # classification
        loss_g_cls = self.classification_loss(pred_fake_cls, c_org) * self.args.lambda_cls_G
        # self recon
        loss_g_self = self.l1_loss(img, img_self) * self.args.lambda_rec
        # cross-cycle recon
        loss_g_cc = self.l1_loss(img, img_recon) * self.args.lambda_rec
        # perceptual loss
        if self.args.vgg_loss is not None:
            img_fake = torch.cat((img_ab, img_ba), dim=0)
            loss_g_p = self.perceptual_loss(img, img_fake) * self.args.lambda_perceptual
        # KL loss - z_c
        loss_kl_zc = self._l2_regularize(z_c) * 0.01
        # KL loss - z_s
        kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss_kl_zs = torch.sum(kl_element).mul_(-0.5) * 0.01
        # total loss G
        loss_g = loss_g_adv + loss_g_cls + loss_g_self + loss_g_cc + loss_kl_zc + loss_kl_zs
        if self.args.use_dis_content:
            loss_g += loss_g_content
        if self.args.vgg_loss is not None:
            loss_g += loss_g_p
            self.loss.g_p = loss_g_p.item()
        loss_g.backward(retain_graph=True)

        self.loss.g_adv = loss_g_adv.item()
        self.loss.g_cls = loss_g_cls.item()
        if self.args.use_dis_content:
            self.loss.g_content = loss_g_content.item()
        self.loss.kl_zc = loss_kl_zc.item()
        self.loss.kl_zs = loss_kl_zs.item()
        self.loss.l1_self_rec = loss_g_self.item()
        self.loss.l1_cc_rec = loss_g_cc.item()
        self.loss.total_g = loss_g.item()

    def backward_content_discriminator(self, z_c):
        pred = self.model.content_discriminator(z_c)
        loss_g_content = self.classification_loss(pred, 1 - self.c_org)
        return loss_g_content

    def backward_decoder_random(self, img, c_org):
        # construct images
        cls_a, cls_b = torch.split(c_org, self.args.batch_size, dim=0)
        z_c = self.model.content_encoder(img)
        z_ca, z_cb = torch.split(z_c, self.args.batch_size, dim=0)
        z_sr = self.get_z_random(c_org.size(0), self.args.latent_dim)
        z_r = self.model.rand_style_encoder(z_sr, c_org)
        z_ra, z_rb = torch.split(z_r, self.args.batch_size, dim=0)
        content = torch.cat([z_cb, z_ca], dim=0)
        style = torch.cat([z_ra, z_rb], dim=0)
        img_random = self.model.decoder(content, style)
        img_br, img_ar = torch.split(img_random, self.args.batch_size, dim=0)
        # Ladv for generator
        pred_fake, pred_fake_cls = self.model.discriminator2(img_random)
        loss_g_adv = self.gan_loss(pred_fake, 1)
        # classification
        loss_g_cls = self.classification_loss(pred_fake_cls, c_org) * self.args.lambda_cls_G
        # latent regression loss
        _, mu2, _= self.model.style_encoder(img_random, c_org)
        mu2_a, mu2_b = torch.split(mu2, self.args.batch_size, dim=0)
        loss_z_l1_a = self.l1_loss(mu2_a, z_ra)
        loss_z_l1_b = self.l1_loss(mu2_b, z_rb)
        loss_z_l1 = (loss_z_l1_a + loss_z_l1_b) * 10
        # perceptual
        if self.args.vgg_loss is not None:
            img_random = torch.cat((img_ar, img_br), dim=0)
            loss_g_p =  self.perceptual_loss(img, img_random) * self.args.lambda_perceptual
        # total G loss
        loss_g = loss_z_l1 + loss_g_adv + loss_g_cls
        if self.args.vgg_loss is not None:
            loss_g += loss_g_p
            self.loss.g_p2 = loss_g_p.item()
        loss_g.backward()
        self.loss.l1_recon_z = loss_z_l1.item()
        self.loss.gan2 = loss_g_adv.item()
        self.loss.gan2_cls = loss_g_cls.item()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def compute_visuals(self):
        img_fake, img_random, img_self = self.forward(self.img, self.c_org)
        img_fake_a, img_fake_b = torch.split(img_fake, self.args.batch_size, dim=0)
        img_random_a, img_random_b = torch.split(img_random, self.args.batch_size, dim=0)
        img_self_a, img_self_b = torch.split(img_self, self.args.batch_size, dim=0)
        images_a = self.normalize_image(self.img_a)
        images_b = self.normalize_image(self.img_b)
        images_a1 = self.normalize_image(img_fake_a)
        images_b1 = self.normalize_image(img_fake_b)
        images_a2 = self.normalize_image(img_random_a)
        images_b2 = self.normalize_image(img_random_b)
        images_a3 = self.normalize_image(img_self_a)
        images_b3 = self.normalize_image(img_self_b)
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a3[0:1, ::]), dim=3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b3[0:1, ::]), dim=3)
        return torch.cat((row1,row2), dim=2)

    def normalize_image(self, x):
        return x[:,0:3,:,:]

    def optimize_parameters(self, global_iter):
        if self.args.use_dis_content:
            if global_iter % self.args.d_iter != 0:
                self.update_content_discriminator(self.img, self.c_org)
            else:
                self.update_discriminator(self.img, self.c_org)
                self.update_generator(self.img, self.c_org)
        else:
            self.update_discriminator(self.img, self.c_org)
            self.update_generator(self.img, self.c_org)