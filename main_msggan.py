"""
To run this template just do:
python generative_adversarial_net.py
After a few epochs, launch TensorBoard to see the images being generated at every batch:
tensorboard --logdir default
"""
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        pass

    def forward(self, z):
        pass


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        pass

    def forward(self, img):
        pass


class MSGGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        pass 

    def forward(self, z):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        pass 

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        pass 

    def on_epoch_end(self):
        pass 


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

    hparams = parser.parse_args()

    main(hparams)