"""
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
import torchvision

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from sklearn.utils import resample

DEVICES = [torch.device('cuda:0'), torch.device('cuda:1')]
WIDTH, HEIGHT = 256, 256
UPSAMPLE = 7  # number of upsamplings
depth = 8
z_dimension = 20
s = 2 ** UPSAMPLE
assert WIDTH % s == 0 and HEIGHT % s == 0
initial_size = (HEIGHT // s, WIDTH // s)

# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super().__init__()
#         pass

#     def forward(self, z):
#         pass


# class Discriminator(nn.Module):
#     def __init__(self, img_shape):
#         super().__init__()
#         pass

#     def forward(self, img):
#         pass

from networks import *

def accumulate(model_accumulator, model, decay=0.993):

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)

class MSGGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        generator = Generator(z_dimension, initial_size, UPSAMPLE, depth)
        discriminator = Discriminator(initial_size, UPSAMPLE, depth)
        score_predictor = FinalDiscriminatorBlock(discriminator.out_channels, initial_size)
        discriminator = nn.Sequential(discriminator, score_predictor)

        self.generator = deepcopy(generator)
        self.discriminator = deepcopy(discriminator)
        self.generator_ema = deepcopy(generator)
        pass 

    def get_noise(self, b):
        z = torch.randn(b, z_dimension)
        z = z / z.norm(p=2, dim=1, keepdim=True)
        return z

    def forward(self, z):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch
        batchs = images.shape[0] #self.hparams.batch

        downsampled = [images]
        for _ in range(UPSAMPLE):
            images = F.avg_pool2d(images, 2)
            downsampled.append(images)

        # from lowest to biggest resolution
        real_images = downsampled[::-1]

        z = self.get_noise(batchs).type_as(images)  # shape [b, z_dimension]
        fake_images = self.generator(z)
        fake_images_detached = [x.detach() for x in fake_images]
        if batch_idx % 100 ==0:
            num = 8
            self.fake_images = fake_images
            self.real_images = real_images

            for k, viz in enumerate(self.fake_images):
                grid = torchvision.utils.make_grid(viz[:num], normalize=True, nrow=4)
                self.logger.experiment.add_image(f'fake/scale_{k}', grid, self.current_epoch)
            for k, viz in enumerate(self.real_images):
                grid = torchvision.utils.make_grid(viz[:num], normalize=True, nrow=4)
                self.logger.experiment.add_image(f'real/scale_{k}', grid, self.current_epoch)

        if optimizer_idx == 1:
            # self.discriminator.requires_grad_(True)
            real_scores = self.discriminator(real_images)
            fake_scores = self.discriminator(fake_images_detached)
            # they have shape [b/2] (because of pacgan)

            r = real_scores - fake_scores.mean()
            f = fake_scores - real_scores.mean()
            discriminator_loss = F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()

            # g = grad(real_scores.sum(), images, create_graph=True)[0]
            # # it has shape [b, 3, h, w]
            # R1 = 0.5 * g.view(b, -1).pow(2).sum(1).mean(0)
            # d_loss = discriminator_loss + R1
            
            d_loss = discriminator_loss
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 0:
            # self.discriminator.requires_grad_(False)
            real_images_detached = [x.detach() for x in real_images]

            real_scores = self.discriminator(real_images_detached)
            fake_scores = self.discriminator(fake_images)
            # they have shape [b/2] (because of pacgan)

            r = real_scores - fake_scores.mean()
            f = fake_scores - real_scores.mean()
            generator_loss = F.relu(1.0 + r).mean() + F.relu(1.0 - f).mean()
            g_loss = generator_loss
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            accumulate(self.generator_ema, self.generator)
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


    def on_epoch_end(self):
        pass

    def train_val_dataloader(self):
        extra = [*zoom_crop(scale=(0.5, 1.3), p=0.5), 
                 *rand_resize_crop(self.hparams.shape, max_scale=1.3),
                 squish(scale=(0.9, 1.2), p=0.5),
                 tilt(direction=(0, 3), magnitude=(-0.3, 0.3), p=0.5),
                 # cutout(n_holes=(1, 5), length=(10, 30), p=0.1)
                 ]
        transforms = get_transforms(max_rotate=11, max_zoom=1.3, max_lighting=0.1, do_flip=False, 
                                    max_warp=0.15, p_affine=0.5, p_lighting=0.3, xtra_tfms=extra)
        transforms = list(transforms); 
        transforms[1] = []

        df=pd.read_csv(os.path.join(self.hparams.data, 'covid_train_v5.csv'))
        df['Non_Covid'] = 1 - df['Covid']
        balancing='up'
        if balancing == 'up':
            df_majority = df[df[self.hparams.pathology]==0]
            df_minority = df[df[self.hparams.pathology]==1]
            print(df_majority[self.hparams.pathology].value_counts())
            df_minority_upsampled = resample(df_minority,
                                     replace=True,     # sample with replacement
                                     n_samples=df_majority[self.hparams.pathology].value_counts()[0],    # to match majority class
                                     random_state=hparams.seed)# reproducible results

            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            df = df_upsampled

        if self.hparams.types == 4:
            dset = (
                ImageList.from_df(df=df, path=os.path.join(self.hparams.data, 'data'), cols='Images')
                .split_by_rand_pct(0.0, seed=self.hparams.seed)
                .label_from_df(cols=['Covid', 'Airspace_Opacity', 'Consolidation', 'Pneumonia'], label_cls=MultiCategoryList)
                .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
                .databunch(bs=self.hparams.batch, num_workers=8)
                .normalize(imagenet_stats)
            )
            return dset.train_dl.dl, dset.valid_dl.dl
        elif self.hparams.types == 2:
            dset = (
                ImageList.from_df(df=df, path=os.path.join(self.hparams.data, 'data'), cols='Images')
                .split_by_rand_pct(0.0, seed=self.hparams.seed)
                .label_from_df(cols=['Covid', 'Non_Covid'], label_cls=MultiCategoryList)
                .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
                .databunch(bs=self.hparams.batch, num_workers=8)
                .normalize(imagenet_stats)
            )
            return dset.train_dl.dl, dset.valid_dl.dl
        pass

    def train_dataloader(self):
        ds_train, ds_valid = self.train_val_dataloader()
        return ds_train


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = MSGGAN(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer()

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=20, help="dimensionality of the latent space")

    # hparams = parser.parse_args()
    # parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=512)

    parser.add_argument('--data', metavar='DIR', default=".", type=str)
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str)
    parser.add_argument('--info', metavar='DIR', default="train_log")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--note', type=str, default="MSGGAN") # Regular, warmup, pretrained
    parser.add_argument('--case', type=str, default="origin")
    # Inference params
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--pred', action='store_true')
    parser.add_argument('--eval', action='store_true')

    # Dataset params
    parser.add_argument("--fast_dev_run", action='store_true')
    parser.add_argument("--percent_check", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument('--Lambda', type=float, default=1.0)
    # parser.add_argument("--e1", type=int, default=0)
    # parser.add_argument("--e2", type=int, default=10)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--types', type=int, default=4)
    parser.add_argument('--shape', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pathology', default='All')
    hparams = parser.parse_args()

    main(hparams)