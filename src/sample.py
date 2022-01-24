import os
import torch

from torchvision import transforms
from PIL import Image
from arguments import TestArguments
from utils import TimerBlock, save_images
from dataset import ImageList
from dataset import VideoDataset
from videoreaders import FrameWriter
from videoreaders import SVOReader

DOMAIN_MAP = ['cloud', 'fog', 'rain', 'sun']

class Sampler(object):
    '''Applies the model to a sample set of images or a video'''
    def __init__(self):
        self.dataset_type = None
        self.transforms = self.get_transforms()

    def load_image_dataset(self, args):
        with TimerBlock('Loading image dataset') as block:
            dataset = ImageList(args.dataroot, transform=self.transforms)
            block.log('Creating dataloader')
            return dataset

    def load_video_dataset(self, args):
        with TimerBlock('Loading data') as block:
            block.log(f"Loading data from {args.dataroot}")
            dataset = VideoDataset(args.dataroot, transform=self.transforms)
            return dataset

    def load_dataset(self, args):
        with TimerBlock("Loading Dataset") as block:
            if os.path.isdir(args.dataroot):
                block.log('Load image dataset')
                dataset = self.load_image_dataset(args)
            else:
                block.log('Load video dataset')
                dataset = self.load_video_dataset(args)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,
                                                     drop_last=True)
        return dataloader

    def get_transforms(self):
        transform = [transforms.Resize((300,300))]
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        return transforms.Compose(transform)

    def load_model(self, args):
        with TimerBlock('Creating model') as block:
            model = args.model(args)
            block.log('Initialize model')
            model.initialize()
            if args.resume:
                block.log('Load pretrained weights')
                model.load(args.resume)
            return model, model.device

    def load_image(self, args, img, device):
        img = Image.open(img).convert('RGB')
        img = self.transforms(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
            img = img.repeat((args.batch_size, 1, 1, 1))
        return img.to(device)

    def load_target(self, args, trg, device):
        """loads one hot target"""
        onehot = torch.zeros((args.batch_size, args.num_domains))
        onehot[:, torch.tensor(trg, dtype=torch.long)] = 1
        return onehot.to(device)

    @torch.no_grad()
    def sample_batch(self, args, model, batch, trg, ref=None, z_sr=None, device=torch.device('cpu')):
        # load target tensor
        trg_t = self.load_target(args, trg, device)
        # load the reference image specific to target
        if ref is not None:
            ref = self.load_image(args, ref, device)
            imgs = model.forward_reference(batch, ref, trg_t)
        elif z_sr is not None:
            imgs = model.forward_random(batch, z_sr, trg_t)
        else:
            raise ValueError('One of ref or z_sr values has to be provided.')
        return imgs

    def sample(self, args, model, dataloader, trgs=None, refs=None, device=torch.device('cpu')):
        with TimerBlock('Running model') as block:
            # get random style vector
            z_sr = model.get_z_random(args.batch_size, args.latent_dim)
            # set targets to all possible targets for random generation
            if trgs is None:
                trgs = range(args.num_domains)
            # check if the refs are provided
            if refs is not None and trgs is not None:
                assert len(trgs) == len(refs), "target and reference should match the shape"
            # loop over all the targets and all the images
            for trg in trgs:
                for i, batch in enumerate(dataloader):
                    if refs is not None:
                        ref = refs[trg]
                        imgs = self.sample_batch(args, model, batch, trg, ref, device=device)
                    else:
                        imgs = self.sample_batch(args, model, batch, trg, z_sr=z_sr, device=device)
                    names = [os.path.join(args.display_dir, str(trg), f'image{i}_{j}.jpg') for j in range(len(imgs))]
                    save_images(imgs, names)

    def run(self):
        with TimerBlock('Starting sampling') as block:
            # load arguments
            args = TestArguments().parse()
            # load model
            model, device = self.load_model(args)
            # load dataset
            dataloader = self.load_dataset(args)
            # run sample method
            args.targets = [DOMAIN_MAP.index(t) for t in args.targets]
            self.sample(args, model, dataloader, args.targets, args.reference, device)

if __name__ == "__main__":
    sampler = Sampler()
    sampler.run()