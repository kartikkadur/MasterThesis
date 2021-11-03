import torch
import itertools

from torch.autograd import Variable
from model.networks import init_net
from utils.tools import tensor_to_image, mask_to_heatmap, overlay, save_image
from .base_model import Model
from . import networks
from . import loss
from PIL import Image
from torchvision import transforms


class SPAGAN(Model):
    """
    Implementation of AttentionGAN model
    """
    @staticmethod
    def modify_commandline_arguments(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--n_updates', type=int, default=5, help='n discriminator updates per single generator update')
        parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss')
        parser.add_argument('--lambda_idt', type=float, default=0.5, help='weight for identity loss')
        parser.add_argument('--lambda_cls_D', type=float, default=1.0, help='classification loss for Discriminator')
        parser.add_argument('--lambda_cls_G', type=float, default=5.0, help='classification loss for Generator')
        return parser

    def __init__(self, args):
        super(SPAGAN, self).__init__(args)
        # loss names
        self.print_losses = ['g', 'cycle', 'cls', 'feat']
        self.visuals = ['real_src', 'real_ref', 'fake', 'rec', 'src_att']
        # define generator
        #self.models.g = networks.UnetGenerator(args.input_nc, args.output_nc, args.ngf, norm_layer=args.norm_layer, input_size=args.crop_size, num_domains=args.num_domains, feature=True)
        self.models.g = networks.ResnetGenerator(args.input_nc, args.output_nc, args.ngf, n_downsampling=2, n_blocks=9, 
                                                 norm_layer=args.norm_layer, num_domains=args.num_domains, feature=True, use_dropout=True)
        # define discriminator
        if self.isTrain:
            self.models.d = networks.NLayerDiscriminator(args.input_nc, args.ndf, norm_layer=args.norm_layer, input_size=args.crop_size, n_domain=args.num_domains)
            self.models.d_c = networks.AttContentDiscriminator(args.input_nc, args.ndf, norm_layer=args.norm_layer, input_size=args.crop_size, n_domain=args.num_domains)
            # define loss functions
            self.criterion.gan = loss.GANLoss(args.gan_mode).to(self.device)
            self.criterion.cycle = torch.nn.L1Loss()
            self.criterion.idt = torch.nn.L1Loss()
            self.criterion.feature = torch.nn.L1Loss()
            self.criterion.classification = torch.nn.BCEWithLogitsLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer.g = torch.optim.Adam(self.models.g.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
            self.optimizer.d = torch.optim.Adam(self.models.d.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
            self.optimizer.d_c = torch.optim.Adam(self.models.d_c.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
            # compile model
            super(SPAGAN, self).compile()
        # init net
        for net in self.models:
            self.models[net] = init_net(self.models[net], args.init_type, args.init_gain, args.gpu_ids)

    def set_inputs(self, inputs):
        self.real_src = inputs['x1'].to(self.device)
        self.real_ref = inputs['x2'].to(self.device)
        self.cls_src = inputs['y1'].to(self.device)
        self.cls_ref = inputs['y2'].to(self.device)

    def forward(self, x, y):
        pred_cls, att = self.models.d_c(x)
        fake, _ = self.models.g(x * att, y)
        return fake

    def backward_D(self):
        # loss with real images
        pred_real, cls_real = self.models.d(self.real_ref)
        loss_d_real = self.criterion.gan(pred_real, True)
        loss_cls = self.criterion.classification(cls_real, self.cls_ref)
        # Fake
        pred_cls, att = self.models.d_c(self.real_src)
        fake, _ = self.models.g(self.real_src * att, self.cls_ref)
        pred_fake, cls_fake = self.models.d(fake.detach())
        loss_d_fake = self.criterion.gan(pred_fake, False)
        # add losses
        loss_d = (loss_d_real + loss_d_fake) * 0.5 + loss_cls * self.args.lambda_cls_D
        loss_d.backward()
        # content loss backward
        loss_cls_content = self.criterion.classification(pred_cls, self.cls_src)
        loss_cls_content.backward()
        # assign loss values
        self.loss.d = loss_d.item()
        self.loss.content = loss_cls_content.item()

    def backward_G(self):
        # identity loss
        fake, feat = self.models.g(self.real_src, self.cls_src)
        self.loss.idt = self.criterion.idt(self.real_src, fake) * self.args.lambda_idt
        _, att = self.models.d_c(self.real_src)
        self.fake, real_feat = self.models.g(self.real_src * att, self.cls_ref)
        d_out, cls_fake = self.models.d(self.fake.detach())
        # for fake images update generator
        self.loss.g = self.criterion.gan(d_out, True)
        self.loss.cls = self.criterion.classification(cls_fake, self.cls_ref) * self.args.lambda_cls_G
        # cycle consistency loss
        self.rec, rec_feat = self.models.g(self.fake, self.cls_src)
        self.loss.cycle = self.criterion.cycle(self.rec, self.real_src) * self.args.lambda_cycle
        # feature loss
        self.loss.feat = self.criterion.feature(real_feat, rec_feat)
        # combined loss and calculate gradients
        loss_g = self.loss.g + self.loss.idt + self.loss.cycle + self.loss.cls + self.loss.feat
        loss_g.backward()

    def optimize_G(self):
        self.set_requires_grad(self.models.d, self.models.d_c, grad=False)
        self._zero_grad()
        self.backward_G()
        self.optimizer.g.step()

    def optimize_D(self):
        self.set_requires_grad(self.models.d, self.models.d_c, grad=True)
        self._zero_grad()
        # discriminator
        self.backward_D()
        self.optimizer.d.step()
        self.optimizer.d_c.step()

    def optimize_parameters(self, *args):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # optimizer discriminator
        self.optimize_D()
        # optimizer generator
        if args[0] % self.args.n_updates == 0:
            self.optimize_G()

    def compute_visuals(self):
        att_src = mask_to_heatmap(self.att_src.data)
        att_ref = mask_to_heatmap(self.att_ref.data)
        self.src_att = overlay(self.real_src, att_src)
        self.ref_att = overlay(self.real_ref, att_ref)

    def predict(self, x, checkpoint=None, a2b = True, img_path=None):
        # prepare image
        img = Image.open(x).convert('RGB')
        transform = transforms.Compose([transforms.Resize((256,256), Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        img = transform(img).unsqueeze(0).to(self.device)
        # create model
        if a2b:
            gen = 'netG_A'
            dis = 'netD_B'
        else:
            gen = 'netG_B'
            dis = 'netD_A'
        # forward operation
        _, dis_out = self.models[dis](img)
        gen_out, _, _ = self.models[gen](dis_out*img)
        out = tensor_to_image(gen_out)
        if img_path:
            save_image(out, img_path)
        return out
