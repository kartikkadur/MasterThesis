import torch
import itertools

from torch.autograd import Variable
from model.networks import init_net
from utils.tools import ImagePool
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
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_cls_D', type=float, default=1.0, help='classification loss for Discriminator')
            parser.add_argument('--lambda_cls_G', type=float, default=5.0, help='classification loss for Generator')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, args):
        """
        """
        super(SPAGAN, self).__init__(args)
        # loss names
        self.print_losses = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visuals = ['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B']
        # define generators
        self.models.netG_A = networks.define_G(args)
        self.models.netG_B = networks.define_G(args)
        # define discriminators
        if self.isTrain:
            self.models.netD_A = networks.define_D(args)
            self.models.netD_B = networks.define_D(args)

        if self.isTrain:
            if args.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(args.input_nc == args.output_nc)
            self.fake_A_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterion.gan = loss.GANLoss(args.gan_mode).to(self.device)
            self.criterion.cycle = torch.nn.L1Loss()
            self.criterion.idt = torch.nn.L1Loss()
            self.criterion.feature = torch.nn.L1Loss()
            self.criterion.classification = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer.G = torch.optim.Adam(itertools.chain(self.models.netG_A.parameters(), self.models.netG_B.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
            self.optimizer.D = torch.optim.Adam(itertools.chain(self.models.netD_A.parameters(), self.models.netD_B.parameters()), lr=args.lr, betas=(args.beta1, 0.999))
            # compile model
            super(SPAGAN, self).compile(loss_names=self.print_losses)

    def set_inputs(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.args.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.cls_real_A = input['A_class' if AtoB else 'B_class'].to(self.device)
        self.cls_real_B = input['B_class' if AtoB else 'A_class'].to(self.device)
        self.image_paths  = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get attention maps
        _, self.att_real_A, self.pred_cls_real_A = self.models.netD_B(self.real_A)
        _, self.att_real_B, self.pred_cls_real_B = self.models.netD_A(self.real_B)
        # run generators
        self.fake_B, _, self.real_feat_A = self.models.netG_A(self.att_real_A * self.real_A)  # G_A(A)
        self.rec_A, _, self.rec_feat_A = self.models.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A, _, self.real_feat_B = self.models.netG_B(self.att_real_B * self.real_B)  # G_B(B)
        self.rec_B, _, self.rec_feat_B = self.models.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake, real_cls):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, _, pred_real_cls = netD(real)
        loss_D_real = self.criterion.gan(pred_real, True)
        # Fake
        pred_fake, _, pred_fake_cls = netD(fake.detach())
        loss_D_fake = self.criterion.gan(pred_fake, False)
        # classification loss
        loss_D_cls = self.criterion.classification(pred_real_cls, real_cls)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + loss_D_cls * self.args.lambda_cls_D
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss.D_A = self.backward_D_basic(self.models.netD_A, self.real_B, fake_B, self.cls_real_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss.D_B = self.backward_D_basic(self.models.netD_B, self.real_A, fake_A, self.cls_real_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.args.lambda_identity
        lambda_A = self.args.lambda_A
        lambda_B = self.args.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A, _, _ = self.models.netG_A(self.real_B)
            self.loss.idt_A = self.criterion.idt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B, _, _ = self.models.netG_B(self.real_A)
            self.loss.idt_B = self.criterion.idt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss.idt_A = 0
            self.loss.idt_B = 0

        d_out_A, _, cls_fake_B = self.models.netD_A(self.fake_B)
        d_out_B, _, cls_fake_A = self.models.netD_B(self.fake_A)
        # classification loss
        loss_cls_A = self.criterion.classification(cls_fake_A, self.cls_real_A)
        loss_cls_B = self.criterion.classification(cls_fake_B, self.cls_real_B)
        self.loss.classification = loss_cls_A + loss_cls_B
        # GAN loss D_A(G_A(A))
        self.loss.G_A = self.criterion.gan(d_out_A, True)
        # GAN loss D_B(G_B(B))
        self.loss.G_B = self.criterion.gan(d_out_B, True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss.cycle_A = self.criterion.cycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss.cycle_B = self.criterion.cycle(self.rec_B, self.real_B) * lambda_B
        # total
        self.loss.cycle = (self.loss.cycle_A.val + self.loss.cycle_B.val)
        # feature losses
        self.loss.feature_A = self.criterion.feature(self.real_feat_A, self.rec_feat_A)
        self.loss.feature_B = self.criterion.feature(self.real_feat_B, self.rec_feat_B)
        self.loss.feature = self.loss.feature_A + self.loss.feature_B
        # combined loss and calculate gradients
        self.loss_G = self.loss.G_A.val + self.loss.G_B.val + \
                      self.loss.cycle + self.loss.feature + \
                      self.loss.idt_A.val + self.loss.idt_B.val +\
                      self.loss.classification
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.models.netD_A, self.models.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer.G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer.G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.models.netD_A, self.models.netD_B], True)
        self.optimizer.D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer.D.step()  # update D_A and D_B's weights

    def compute_visuals(self):
        attn_real_A = mask_to_heatmap(self.att_real_A.data)
        attn_real_B = mask_to_heatmap(self.att_real_B.data)
        attn_fake_A = mask_to_heatmap(self.att_rec_A.data)
        attn_fake_B = mask_to_heatmap(self.att_rec_B.data)
        self.attn_real_A = overlay(self.real_A, attn_real_A)
        self.attn_real_B = overlay(self.real_B, attn_real_B)
        self.attn_fake_A = overlay(self.fake_A, attn_fake_A)
        self.attn_fake_B = overlay(self.fake_B, attn_fake_B)

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
