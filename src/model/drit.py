import torch

from torch import nn
from model.networks import init_net
from utils.tools import tensor_to_image, mask_to_heatmap, overlay, save_image
from .base_model import Model
from . import networks
from . import loss
from PIL import Image


class DRIT(Model):
    @staticmethod
    def modify_commandline_arguments(parser, is_train=True):
        parser.set_defaults(no_dropout=False)
        parser.add_argument('--n_updates', type=int, default=3, help='n discriminator updates per single generator update')

        parser.add_argument('--lambda_cls', type=float, default=1.0, help='classification loss for Discriminator')
        parser.add_argument('--lambda_cls_G', type=float, default=5.0, help='classification loss for Generator')
        parser.add_argument('--latent_dim', type=int, default=16, help='latent dim size')
        parser.add_argument('--concat', action='store_true')
        parser.add_argument('--d_content', action='store_true')
        parser.add_argument('--spectral_norm', action='store_true')
        parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
        parser.add_argument('--lr_dcontent', type=float, default=1e-6, help='lr for content discriminator')
        return parser

    def __init__(self, args):
        super(DRIT, self).__init__(args)
        torch.autograd.set_detect_anomaly(True)
        self.visuals = ['final']
        self.models.dis1 = networks.DomainDiscriminator(args.input_nc, norm_layer=args.norm_layer, sn=args.spectral_norm, num_domains=args.num_domains, image_size=args.crop_size)
        self.models.dis2 = networks.DomainDiscriminator(args.input_nc, norm_layer=args.norm_layer, sn=args.spectral_norm, num_domains=args.num_domains, image_size=args.crop_size)
        self.models.enc_c = networks.ContentEncoder(args.input_nc)
        if self.args.concat:
            self.models.enc_a = networks.AttributeEncoderConcat(args.input_nc, output_nc=self.args.latent_dim, num_domains=args.num_domains, \
                                        norm_layer=None, activation='leaky_relu')
            self.models.gen = networks.DomainGeneratorConcat(args.input_nc, num_domains=args.num_domains, latent_dim=args.latent_dim)
        else:
            self.models.enc_a = networks.AttributeEncoder(args.input_nc, output_nc=args.latent_dim, num_domains=args.num_domains)
            self.models.gen = networks.DomainGenerator(args.input_nc, latent_dim=args.latent_dim, num_domains=args.num_domains)

        for net in self.models:
            self.optimizer[net] = torch.optim.Adam(self.models[net].parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        # content discriminator
        self.models.dis_c = networks.ContentDiscriminator(num_domains=args.num_domains) 
        self.optimizer.dis_c = torch.optim.Adam(self.models.dis_c.parameters(), lr=args.lr_dcontent, betas=(0.5, 0.999), weight_decay=args.weight_decay)

        self.criterion.classification = nn.BCEWithLogitsLoss()
        self.initialize()
        super(DRIT, self).compile()

    def initialize(self):
        for net in self.models:
            self.models[net] = init_net(self.models[net], self.args.init_type, self.args.init_gain, self.args.gpu_ids)

    def set_inputs(self, inputs):
        self.x_a = inputs['x1'].to(self.device)
        self.x_b = inputs['x2'].to(self.device)
        self.x = torch.cat((self.x_a, self.x_b), dim=0)
        self.y_a = inputs['y1'].to(self.device)
        self.y_b = inputs['y2'].to(self.device)
        self.y = torch.cat((self.y_a, self.y_b), dim=0)

    def concat(self, a, b, dim=0):
        return torch.cat((a,b), dim=dim)

    def split(self, a, size, dim=0):
        return torch.split(a, size, dim)

    def get_z_random(self, batchSize, latent_dim):
        z = torch.randn(batchSize, latent_dim).to(self.device)
        return z

    def forward_content(self, inputs):
        """encodes content representations"""
        z_content = self.models.enc_c(inputs)
        return z_content

    def forward_attribute(self, inputs, c_org):
        """returns encoded attributes from the image"""
        if self.args.concat:
            mu, logvar = self.models.enc_a(inputs, c_org)
            std = logvar.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1))
            z_attr = eps.mul(std).add_(mu)
        else:
            mu = 0
            logvar = 0
            z_attr = self.models.enc_a(inputs, c_org)
        return z_attr, mu, logvar

    def forward_generator(self, cont_a, cont_b, att_a, att_b, y_a, y_b):
        fake1 = self.models.gen(cont_b, att_a, y_a)
        fake2 = self.models.gen(cont_a, att_b, y_b)
        return fake1, fake2

    def forward_generator_all(self, cont_a, cont_b, att_a, att_b, random, y_a, y_b):
        fake_a, fake_b = self.forward_generator(cont_a, cont_b, att_a, att_b, y_a, y_b)
        fake_a_random, fake_b_random = self.forward_generator(cont_a, cont_b, random, random, y_a, y_b)
        fake_aa, fake_bb = self.forward_generator(cont_b, cont_a, att_a, att_b, y_a, y_b)
        return fake_a, fake_b, fake_a_random, fake_b_random, fake_aa, fake_bb

    def forward_discriminator(self, netD, inputs):
        pred_fake, pred_cls = netD(inputs)
        return pred_fake, pred_cls

    def forward_content_discriminator(self, z):
        pred_cls = self.models.dis_c(z.detach())
        return pred_cls

    def forward(self):
        # get encoded z_c
        self.z_content = self.forward_content(self.x)
        self.z_content_a, self.z_content_b = torch.split(self.z_content, self.args.batch_size, dim=0)
        # get encoded z_a
        self.z_attr, self.mu, self.logvar = self.forward_attribute(self.x, self.y)
        self.z_attr_a, self.z_attr_b = torch.split(self.z_attr, self.args.batch_size, dim=0)
        # get random z_a
        self.z_random = self.get_z_random(self.args.batch_size, self.args.latent_dim)
        # get fake generated images
        self.fake_a, self.fake_b, self.fake_a_random, self.fake_b_random, self.fake_aa, self.fake_bb = self.forward_generator_all(
                                                self.z_content_a, self.z_content_b, self.z_attr_a, self.z_attr_b, self.z_random, self.y_a, self.y_b)
        # get reconstructed encoded z_c
        self.fake = torch.cat((self.fake_a, self.fake_b), dim=0)
        self.fake_random = torch.cat((self.fake_a_random, self.fake_b_random), dim=0)
        self.z_content_recon = self.forward_content(self.fake)
        self.z_content_recon_a, self.z_content_recon_b = torch.split(self.z_content_recon, self.args.batch_size, dim=0)
        # get reconstructed encoded z_a
        self.z_attr_recon, self.mu_recon, self.logvar_recon = self.forward_attribute(self.fake, self.y)
        self.z_attr_recon_a, self.z_attr_recon_b = torch.split(self.z_attr_recon, self.args.batch_size, dim=0)
        # second cross translation
        self.fake_a_recon, self.fake_b_recon = self.forward_generator(self.z_content_recon_b, self.z_content_recon_a,
                                                                      self.z_attr_recon_a, self.z_attr_recon_b, self.y_a, self.y_b)
        # display
        self.image_display = torch.cat((self.x_a[0:1].detach().cpu(), self.fake_b[0:1].detach().cpu(), \
                                        self.fake_b_random[0:1].detach().cpu(), self.fake_aa[0:1].detach().cpu(), self.fake_a_recon[0:1].detach().cpu(), \
                                        self.x_b[0:1].detach().cpu(), self.fake_a[0:1].detach().cpu(), \
                                        self.fake_a_random[0:1].detach().cpu(), self.fake_bb[0:1].detach().cpu(), self.fake_b_recon[0:1].detach().cpu()), dim=0)
        # for latent regression
        self.z_attr_random, self.mu2, self.logvar2 = self.forward_attribute(self.fake_random, self.y)
        self.z_attr_random_a, self.z_attr_random_b = torch.split(self.z_attr_random, self.args.batch_size, dim=0)

    def update_D_content(self):
        self.z_content = self.forward_content(self.x)
        self.optimizer.dis_c.zero_grad()
        pred_cls = self.forward_content_discriminator(self.z_content)
        loss_D_content = self.criterion.classification(pred_cls, self.y)
        loss_D_content.backward()
        self.loss_dis_c = loss_D_content.item()
        nn.utils.clip_grad_norm_(self.models.dis_c.parameters(), 5)
        self.optimizer.dis_c.step()

    def update_D(self):        
        self.optimizer.dis1.zero_grad()
        self.loss_gan_d1, self.loss_cls_d1 = self.backward_D(self.models.dis1, self.x, self.fake)
        self.optimizer.dis1.step()

        self.optimizer.dis2.zero_grad()
        self.loss_gan_d2, self.loss_cls_d2 = self.backward_D(self.models.dis2, self.x, self.fake_random)
        self.optimizer.dis2.step()

    def backward_D(self, netD, real, fake):
        pred_fake, pred_fake_cls = netD(fake.detach())
        pred_real, pred_real_cls = netD(real)
        loss_D_gan = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).to(self.device)
            all1 = torch.ones_like(out_real).to(self.device)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D_gan += ad_true_loss + ad_fake_loss

        loss_D_cls = self.criterion.classification(pred_real_cls, self.y)
        loss_D = loss_D_gan + self.args.lambda_cls * loss_D_cls
        loss_D.backward()
        return loss_D_gan, loss_D_cls

    def update_EG(self):
        # update G, Ec, Ea
        self.optimizer.enc_c.zero_grad()
        self.optimizer.enc_a.zero_grad()
        self.optimizer.gen.zero_grad()
        self.backward_EG()
        self.optimizer.enc_c.step()
        self.optimizer.enc_a.step()
        self.optimizer.gen.step()
        # call forward
        self.forward()
        # update G, Ec
        self.optimizer.enc_c.zero_grad()
        self.optimizer.gen.zero_grad()
        self.backward_G_alone()
        self.optimizer.enc_c.step()
        self.optimizer.gen.step()

    def backward_EG(self):
        # content adv for generator
        if self.args.d_content:
            loss_G_adv_content = self.backward_G_content(self.z_content, self.y)
        # adv for generator
        pred_fake, pred_fake_cls = self.forward_discriminator(self.models.dis1, self.x)
        loss_G_adv = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.device)
            loss_G_adv += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
        # classification
        loss_G_cls = self.criterion.classification(pred_fake_cls, self.y) * self.args.lambda_cls_G
        # self and cross-cycle recon
        loss_G_L1_self = torch.mean(torch.abs(self.x - torch.cat((self.fake_aa, self.fake_bb), dim=0))) * self.args.lambda_rec
        loss_G_L1_cc = torch.mean(torch.abs(self.x - torch.cat((self.fake_a_recon, self.fake_b_recon), dim=0))) * self.args.lambda_rec
        # KL loss - z_c
        loss_kl_zc = self._l2_regularize(self.z_content) * 0.01
        # KL loss - z_a
        if self.args.concat:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            loss_kl_za = torch.sum(kl_element).mul_(-0.5) * 0.01
        else:
            loss_kl_za = self._l2_regularize(self.z_attr) * 0.01

        loss_G = loss_G_adv + loss_G_cls + loss_G_L1_self + loss_G_L1_cc + loss_kl_zc + loss_kl_za
        if self.args.d_content:
            loss_G += loss_G_adv_content
        loss_G.backward(retain_graph=True)

        self.gan_loss = loss_G_adv.item()
        self.gan_cls_loss = loss_G_cls.item()
        if self.args.d_content:
            self.gan_loss_content = loss_G_adv_content.item()
        self.kl_loss_zc = loss_kl_zc.item()
        self.kl_loss_za = loss_kl_za.item()
        self.l1_self_rec_loss = loss_G_L1_self.item()
        self.l1_cc_rec_loss = loss_G_L1_cc.item()
        self.G_loss = loss_G.item()

    def backward_G_content(self, z, c_org):
        pred_cls = self.forward_content_discriminator(z)
        loss_G_content = self.criterion.classification(pred_cls, 1-c_org)
        return loss_G_content

    def backward_G_alone(self):
        # adv for generator
        pred_fake, pred_fake_cls = self.forward_discriminator(self.models.dis2, self.x)
        loss_G_adv2 = 0
        for out_a in pred_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).to(self.device)
            loss_G_adv2 += nn.functional.binary_cross_entropy(outputs_fake, all_ones)

        # classification
        loss_G_cls2 = self.criterion.classification(pred_fake_cls, self.y) * self.args.lambda_cls_G

        # latent regression loss
        if self.args.concat:
            mu2_a, mu2_b = torch.split(self.mu2, self.args.batch_size, dim=0)
            loss_z_L1_a = torch.mean(torch.abs(mu2_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(mu2_b - self.z_random)) * 10
        else:
            loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
            loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

        loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_adv2 + loss_G_cls2
        loss_z_L1.backward()
        self.l1_recon_z_loss = loss_z_L1_a.item() + loss_z_L1_b.item()
        self.gan2_loss = loss_G_adv2.item()
        self.gan2_cls_loss = loss_G_cls2.item()

    def optimize_parameters(self, *args):
        it = args[0]
        self.forward()
        if self.args.d_content:
            if (it + 1) % self.args.n_updates != 0:
                self.update_D_content()
            else:
                self.update_D()
                self.update_EG()
        else:
            self.update_D()
            self.update_EG()

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def compute_visuals(self):
        images_a = self.normalize_image(self.x_a).detach()
        images_b = self.normalize_image(self.x_b).detach()
        images_a1 = self.normalize_image(self.fake_a).detach()
        images_a2 = self.normalize_image(self.fake_a_random).detach()
        images_a3 = self.normalize_image(self.fake_a_recon).detach()
        images_a4 = self.normalize_image(self.fake_aa).detach()
        images_b1 = self.normalize_image(self.fake_b).detach()
        images_b2 = self.normalize_image(self.fake_b_random).detach()
        images_b3 = self.normalize_image(self.fake_b_recon).detach()
        images_b4 = self.normalize_image(self.fake_bb).detach()
        row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
        row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
        self.final = torch.cat((row1,row2),2)

    def normalize_image(self, x):
        return x[:,0:3,:,:]