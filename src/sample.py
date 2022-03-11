import os
import torch
import torchvision

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
        # transforms.Resize((540,960))
        transform = [transforms.Resize((540,960))]
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
            imgs, rt, mu = model.forward_reference(batch, ref, trg_t)
        elif z_sr is not None:
            imgs, rt, mu = model.forward_random(batch, z_sr, trg_t)
        else:
            raise ValueError('One of ref or z_sr values has to be provided.')
        return imgs, rt, mu

    @torch.no_grad()
    def sample(self, args, model, dataloader, trgs=None, refs=None, device=torch.device('cpu')):
        with TimerBlock('Running model') as block:
            # get random style vector
            #z_sr = model.get_z_random(args.batch_size, args.latent_dim)
            # set targets to all possible targets for random generation
            if trgs is None:
                trgs = range(args.num_domains)
            # check if the refs are provided
            if refs is not None and trgs is not None:
                assert len(trgs) == len(refs), "target and reference should match the shape"
            # loop over all the targets and all the images
            for t, trg in enumerate(trgs):
                z_sr = model.get_z_random(args.batch_size, args.latent_dim)
                for i, batch in enumerate(dataloader):
                    if refs is not None:
                        ref = refs[t]
                        imgs, _, _ = self.sample_batch(args, model, batch, trg, ref, device=device)
                    else:
                        imgs, _, _ = self.sample_batch(args, model, batch, trg, z_sr=z_sr, device=device)
                    names = [os.path.join(args.display_dir, str(trg), f'image{t}_{i}_{j}.jpg') for j in range(len(imgs))]
                    save_images(imgs, names)

    @torch.no_grad()
    def sample_diverse(self, args, model, dataloader, trgs=None, refs=None, device=torch.device('cpu')):
        with TimerBlock('Running model') as block:
            # get random style vector
            #z_sr = model.get_z_random(args.batch_size, args.latent_dim)
            # set targets to all possible targets for random generation
            if trgs is None:
                trgs = range(args.num_domains)
            # check if the refs are provided
            if refs is not None and trgs is not None:
                assert len(trgs) == len(refs), "target and reference should match the shape"
            # loop over all the targets and all the images
            for t, trg in enumerate(trgs):
                z_sr = model.get_z_random(args.batch_size, args.latent_dim)
                for i, batch in enumerate(dataloader):
                    if refs is not None:
                        ref = refs[t]
                        imgs, _, _ = self.sample_batch(args, model, batch, trg, ref, device=device)
                    else:
                        imgs, _, _ = self.sample_batch(args, model, batch, trg, z_sr=z_sr, device=device)
                    names = [os.path.join(args.display_dir, str(t), f'{i}.jpg') for j in range(len(imgs))]
                    save_images(imgs, names)

    def generate_image_grid(self, args, model, dataloader, refs=None, trgs=None, device=torch.device('cpu')):
        """
        generates a grid of images
        """
        exetimes = []
        memory = []
        if refs is None:
            z_sr = model.get_z_random(args.batch_size, args.latent_dim)
        if trgs is None:
            trgs = range(args.num_domains)
        if refs is not None:
            assert len(refs) == len(trgs), "Reference for each target class has to be provided"
        rows = []
        cols = []
        if refs is not None:
            rows.append(torch.ones(1, 3, 512, 512).to(device))
            for ref in refs:
                rows.append(self.transforms(Image.open(ref).convert('RGB')).unsqueeze(0).to(device))
            cols.append(torch.cat(rows, dim=3))
            rows = []
        for i, batch in enumerate(dataloader):
            rows = []
            rows.append(batch.to(device))
            for t, trg in enumerate(trgs):
                if refs is not None:
                    ref = refs[t]
                    imgs, exe_time, mem = self.sample_batch(args, model, batch, trg, ref, device=device)
                else:
                    imgs, exe_time, mem = self.sample_batch(args, model, batch, trg, z_sr=z_sr, device=device)
                rows.append(imgs)
                exetimes.append(exe_time)
                memory.append(mem)
            cols.append(torch.cat(rows, dim=3))
        images = torch.cat(cols, dim=2)
        print(f"Avg execution time : {sum(exetimes)/ len(exetimes)}, cuda memory usage: {sum(memory)/len(memory)}")
        torchvision.utils.save_image(images/2.0+0.5, './grid.png', padding=5, pad_value=1)

    def generate_multiple_styles(self, args, model, image, trg, refs=None, n_samples=4, device=torch.device('cpu')):
        """
        generates a grid of images
        """
        images = []
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.transforms(image)
        if refs is not None:
            n_samples = len(refs)
            images.append(torch.ones(1, 3, 512, 512).to(device))
            for ref in refs:
                images.append(self.transforms(Image.open(ref).convert('RGB')).unsqueeze(0).to(device))
        images.append(image.to(device))
        
        for i in range(n_samples):
            if refs is not None:
                ref = refs[i]
                imgs, exe_time, mem = self.sample_batch(args, model, image, trg, ref, device=device)
            else:
                z_sr = model.get_z_random(args.batch_size, args.latent_dim)
                imgs, exe_time, mem = self.sample_batch(args, model, image, trg, z_sr=z_sr, device=device)
            images.append(imgs)
        images = torch.cat(images, dim=0)
        torchvision.utils.save_image(images/2.0+0.5, './grid.png', nrow=n_samples+1, padding=0)

        
    def run(self):
        with TimerBlock('Starting sampling') as block:
            # load arguments
            args = TestArguments().parse()
            # load model
            model, device = self.load_model(args)
            # load dataset
            dataloader = self.load_dataset(args)
            # map targets to class labels
            args.targets = [DOMAIN_MAP.index(t) for t in args.targets]
            if args.gen_grid:
                # make image grid
                block.log("Generating image grid")
                self.generate_image_grid(args, model, dataloader, args.reference, args.targets, device)
            elif args.gen_style:
                block.log("Generating multiple style image grid")
                batch = next(iter(dataloader))
                self.generate_multiple_styles(args, model, batch, args.targets[0], args.reference, device=device)
            else:
                # run sample method
                block.log("Running sample")
                self.sample(args, model, dataloader, args.targets, args.reference, device)
            
if __name__ == "__main__":
    sampler = Sampler()
    sampler.run()