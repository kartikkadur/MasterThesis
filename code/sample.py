import os
import torch

from torchvision import transforms
from PIL import Image
from arguments import TestArguments
from utils import TimerBlock, save_images, tensor_to_image
from dataset import ImageFolder
from videoreaders import FrameReader
from videoreaders import FrameWriter

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
            self.load_image()
            block.log("Get transforms")
            transform = self.get_transforms()
            block.log(f"Saving image into the folder {self.args.display_dir}")
            for i, batch in enumerate(self.dataset):
                img, label = batch
                # apply transforms
                img = transform(img)
                # generate image for each domain
                for domain in range(self.args.num_domains):
                    with torch.no_grad():
                        imgs = self.model.forward_random(img.unsqueeze(0), domain)
                        names = [os.path.join(self.args.display_dir, f'image_{domain}_{i}_{j}.png') for j in range(len(imgs))]
                    save_images(imgs, names)

    def generate_video(self, trg, ref=None):
        with TimerBlock("Generating Video") as block:
            block.log("Loading video dataset")
            self.load_video()
            block.log("Get transforms")
            transform = self.get_transforms()
            block.log(f"Writing video into the directory {self.args.display_dir}")
            z_random_style = self.model.get_z_random(1, self.args.latent_dim)
            with FrameWriter(self.args.display_dir, self.args.vid_fname, self.args.out_fmt) as fr:
                for i, frame in enumerate(self.dataset):
                    frame = Image.fromarray(frame)
                    frame = transform(frame)
                    with torch.no_grad():
                        img = self.model.forward_random(frame.unsqueeze(0), z_random_style, trg)
                    img = tensor_to_image(img)
                    fr.write(img, i)

    def run(self):
        self.create_model()
        if 'image' in self.args.out_fmt:
            self.generate_image()
        else:
            self.generate_video(self.args.trg_cls)

if __name__ == "__main__":
    args = TestArguments().parse()
    sample = Sampler(args)
    sample.run()