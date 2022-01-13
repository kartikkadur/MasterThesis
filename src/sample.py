import os
import torch

from torchvision import transforms
from PIL import Image
from arguments import TestArguments
from utils import TimerBlock, save_images, tensor_to_image
from dataset import ImageFolder
from videoreaders import FrameReader
from videoreaders import FrameWriter
from videoreaders import SVOReader

class Sampler(object):
    '''Applies the model to a sample set of images or a video'''
    def __init__(self, args):
        self.args = args
        self.args.num_domains=5

    def load_image(self):
        with TimerBlock('Loading data') as block:
            self.dataset = ImageFolder(self.args)
            block.log('Create dataloader')

    def load_video(self):
        with TimerBlock('Loading data') as block:
            block.log(f"Loading data from {self.args.dataroot}")
            self.dataset = FrameReader(self.args.dataroot)

    def get_transforms(self):
        transform = [transforms.Resize((self.args.crop_size, self.args.crop_size), Image.BICUBIC)]
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        return transforms.Compose(transform)

    def create_model(self):
        with TimerBlock('Creating model') as block:
            self.model = self.args.model(self.args)
            block.log('Initialize model')
            self.model.initialize()
            if self.args.resume:
                block.log('Load pretrained weights')
                self.model.load(self.args.resume)

    def generate_image(self, trg=None, ref=None):
        with TimerBlock('Generating Images') as block:
            block.log('Load image dataset')
            if os.path.isdir(self.args.dataroot):
                self.load_image()
            else:
                self.load_video()
            block.log("Get transforms")
            transform = self.get_transforms()
            z_random_style = self.model.get_z_random(1, self.args.latent_dim)
            block.log(f"Writing video into the directory: {self.args.display_dir}")
            for i, batch in enumerate(self.dataset):
                if os.path.isdir(self.args.dataroot):
                    img, _ = batch
                else:
                    img = batch
                    img = Image.fromarray(img)
                # apply transforms
                img = transform(img).unsqueeze(0)
                with torch.no_grad():
                    if ref is not None:
                        ref = Image.open(ref).convert('RGB')
                        ref = transform(ref).unsqueeze(0)
                        imgs = self.model.forward_reference(img, ref, trg)
                    else:
                        imgs = self.model.forward_random(img, z_random_style, trg)
                    names = [os.path.join(self.args.display_dir, f'image_{trg}_{i}_{j}.png') for j in range(len(imgs))]
                save_images(imgs, names)

    def generate_video(self, trg, ref=None):
        with TimerBlock("Generating Video") as block:
            block.log("Loading video dataset")
            self.load_video()
            block.log("Get transforms")
            transform = self.get_transforms()
            block.log(f"Writing video into the directory: {self.args.display_dir}")
            z_random_style = self.model.get_z_random(1, self.args.latent_dim)
            with FrameWriter(self.args.display_dir, self.args.vid_fname, self.args.out_fmt) as fr:
                for i, frame in enumerate(self.dataset):
                    frame = Image.fromarray(frame)
                    frame = transform(frame).unsqueeze(0)
                    with torch.no_grad():
                        if ref is not None:
                            ref = Image.open(ref).convert('RGB')
                            ref = transform(ref).unsqueeze(0)
                            img = self.model.forward_reference(frame, ref, trg)
                        else:
                            img = self.model.forward_random(frame, z_random_style, trg)
                    img = tensor_to_image(img)
                    fr.write(img, i)

    def generate_from_svo(self, trg, ref=None):
        with TimerBlock('Generating Images from SVO file') as block:
            block.log("Get transforms")
            transform = self.get_transforms()
            z_random_style = self.model.get_z_random(1, self.args.latent_dim)
            block.log(f"Writing outputs into the directory: {self.args.display_dir}")
            with SVOReader(self.args.dataroot, self.args.display_dir, output=self.args.out_fmt) as self.dataset:
                for i in range(len(self.dataset)):
                    frame = self.dataset.get_frame()
                    frame = Image.fromarray(frame)
                    frame = transform(frame).unsqueeze(0)
                    with torch.no_grad():
                        if ref is not None:
                            ref = Image.open(ref).convert('RGB')
                            ref = transform(ref).unsqueeze(0)
                            img = self.model.forward_reference(frame, ref, trg)
                        else:
                            img = self.model.forward_random(frame, z_random_style, trg)
                    self.dataset.write(img, i)

    def run(self):
        with TimerBlock('Starting sampling') as block:
            self.create_model()
            if self.args.dataroot.endswith('.svo'):
                self.generate_from_svo(self.args.trg_cls)
            elif 'image' in self.args.out_fmt:
                self.generate_image(self.args.trg_cls)
            else:
                self.generate_video(self.args.trg_cls)

if __name__ == "__main__":
    args = TestArguments().parse()
    sampler = Sampler(args)
    sampler.run()