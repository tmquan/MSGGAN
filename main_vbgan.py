from networks import *
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
from pytorch_lightning.callbacks import ModelCheckpoint

from pprint import pprint
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from sklearn.utils import resample

DEVICES = [torch.device('cuda:0'), torch.device('cuda:1')]
WIDTH, HEIGHT = 256, 256
UPSAMPLE = 7  # number of upsamplings
depth = 8
z_dimension = 20
n_dimension = 16
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


def accumulate(model_accumulator, model, decay=0.993):
    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


class MSGGAN(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        generator = Generator(self.hparams.latent_dim+self.hparams.types, initial_size, UPSAMPLE, depth)
        discriminator = Discriminator(initial_size, UPSAMPLE, depth)
        score_predictor = FinalDiscriminatorBlock(discriminator.out_channels, initial_size)
        discriminator = nn.Sequential(discriminator, score_predictor)

        self.generator = deepcopy(generator)
        self.discriminator = deepcopy(discriminator)
        self.generator_ema = deepcopy(generator)

        self.classifier = torchvision.models.densenet121(pretrained=True)
        self.classifier.features.relu0 = nn.LeakyReLU(0.02)
        # print(self.classifier)
        self.classifier.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.hparams.types),  # 5 diseases
            nn.Sigmoid(),
        )

        def replace_relu(m):
            if type(m).__name__ == "_DenseLayer":
                m.relu1 = nn.LeakyReLU(0.02)
                m.relu2 = nn.LeakyReLU(0.02)
        self.classifier.apply(replace_relu)

        pass

    # def get_noise(self, b, types=4):
        # n = torch.randn(b, n_dimension)
        # n = n / n.norm(p=2, dim=1, keepdim=True)
        # p = torch.empty(b, types).random_(2)
        # z = torch.cat([n, p*2-1], dim=1)
        # return z

    def forward(self, z):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, labels = batch
        batchs = images.shape[0]  # self.hparams.batch

        downsampled = [images]
        for _ in range(UPSAMPLE):
            images = F.avg_pool2d(images, 2)
            downsampled.append(images)

        # from lowest to biggest resolution
        real_images = downsampled[::-1]

        # z = self.get_noise(batchs).type_as(images)  # shape [b, z_dimension]
        n = torch.randn(batchs, self.hparams.latent_dim).type_as(images)
        n = n / n.norm(p=2, dim=1, keepdim=True)
        p = torch.empty(batchs, self.hparams.types).random_(2).type_as(images)
        z = torch.cat([n, p * 2 - 1], dim=1)

        fake_images = self.generator(z)
        fake_images_detached = [x.detach() for x in fake_images]

        if batch_idx % 100 == 0:
            num = 8
            self.fake_images = fake_images
            self.real_images = real_images

            for k, viz in enumerate(self.fake_images):
                grid = torchvision.utils.make_grid(viz[:num], normalize=True, nrow=4)
                self.logger.experiment.add_image(f'fake/scale_{k}', grid, self.current_epoch)
            for k, viz in enumerate(self.real_images):
                grid = torchvision.utils.make_grid(viz[:num], normalize=True, nrow=4)
                self.logger.experiment.add_image(f'real/scale_{k}', grid, self.current_epoch)

        # Classification
        fake_p = self.classifier(fake_images_detached[-1])
        bce_fake = nn.BCELoss()(fake_p, p)
        real_p = self.classifier(real_images[-1])
        bce_real = nn.BCELoss()(real_p, labels)
        Lambda = self.hparams.Lambda
        c_loss = bce_real + Lambda * bce_fake


        if optimizer_idx == 2:
            tqdm_dict = {'c_loss': c_loss}
            output = OrderedDict({
                'loss': c_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            # self.discriminator.requires_grad_(True)
            real_scores = self.discriminator(real_images)
            fake_scores = self.discriminator(fake_images)
            # they have shape [b/2] (because of pacgan)

            r = real_scores - fake_scores.mean()
            f = fake_scores - real_scores.mean()
            discriminator_loss = torch.abs(1.0 - r).mean() + torch.abs(1.0 + f).mean()

            # g = grad(real_scores.sum(), images, create_graph=True)[0]
            # # it has shape [b, 3, h, w]
            # R1 = 0.5 * g.view(b, -1).pow(2).sum(1).mean(0)
            # d_loss = discriminator_loss + R1

            d_loss = discriminator_loss
            # d_loss += c_loss
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
            generator_loss = torch.abs(1.0 + r).mean() + torch.abs(1.0 - f).mean()
            g_loss = generator_loss
            # # Calculate w1 and w2
            e1 = 0 #self.hparams.e1
            e2 = 9 #self.hparams.e2
            assert e2 > e1
            ep = self.current_epoch
            if ep < e1:
                w1 = 1
                w2 = 0
            elif ep > e2:
                w1 = 0
                w2 = 1
            else:
                w2 = (ep - e1) / (e2 - e1)
                w1 = (e2 - ep) / (e2 - e1)
            ell1_loss = 0
            for fake_one, one in zip(fake_images, real_images_detached): #torch.mean(torch.abs(fake_imgs - imgs))
                ell1_loss += torch.nn.L1Loss()(fake_one, one)
            g_loss *= w2
            g_loss += w1*ell1_loss

            g_loss += bce_fake
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
        opt_c = torch.optim.Adam(self.classifier.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d, opt_c], []
        # sch_g = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10),
        #          'interval': 'step'}
        # sch_d = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10),
        #          'interval': 'step'}
        # sch_c = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=10),
        #          'interval': 'step'}
        # return [opt_g, opt_d, opt_c], [sch_g, sch_d, sch_c]

    def train_val_dataloader(self):
        extra = [*zoom_crop(scale=(0.9, 1.2), p=0.5),
                 *rand_resize_crop(self.hparams.shape, max_scale=1.3),
                 squish(scale=(0.9, 1.1), p=0.5),
                 tilt(direction=(0, 3), magnitude=(-0.3, 0.3), p=0.5),
                 # cutout(n_holes=(1, 5), length=(10, 30), p=0.1)
                 ]
        transforms = get_transforms(max_rotate=10, max_zoom=1.2, max_lighting=0.2, do_flip=False,
                                    max_warp=0.10, p_affine=.75, p_lighting=.75, xtra_tfms=extra)
        transforms = list(transforms)
        transforms[1] = []

        df = pd.read_csv(os.path.join(self.hparams.data, 'covid_train_v5.csv'))
        df['Non_Covid'] = 1 - df['Covid']
        balancing = 'up'
        if balancing == 'up':
            df_majority = df[df[self.hparams.pathology] == 0]
            df_minority = df[df[self.hparams.pathology] == 1]
            print(df_majority[self.hparams.pathology].value_counts())
            df_minority_upsampled = resample(df_minority,
                                             replace=True,     # sample with replacement
                                             n_samples=df_majority[self.hparams.pathology].value_counts()[
                                                 0],    # to match majority class
                                             random_state=hparams.seed)  # reproducible results

            df_upsampled = pd.concat([df_majority, df_minority_upsampled])
            df = df_upsampled

        if self.hparams.types == 4:
            dset = (
                ImageList.from_df(df=df, path=os.path.join(
                    self.hparams.data, 'data'), cols='Images')
                .split_by_rand_pct(0.0, seed=self.hparams.seed)
                .label_from_df(cols=['Covid', 'Airspace_Opacity', 'Consolidation', 'Pneumonia'], label_cls=MultiCategoryList)
                .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
                .databunch(bs=self.hparams.batch, num_workers=8)
                .normalize(imagenet_stats)
            )
            return dset.train_dl.dl, dset.valid_dl.dl
        elif self.hparams.types == 2:
            dset = (
                ImageList.from_df(df=df, path=os.path.join(
                    self.hparams.data, 'data'), cols='Images')
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

    def eval_dataloader(self):
        extra = [*zoom_crop(scale=(0.9, 1.2), p=0.5),
                 *rand_resize_crop(self.hparams.shape, max_scale=1.3),
                 squish(scale=(0.9, 1.1), p=0.5),
                 tilt(direction=(0, 3), magnitude=(-0.3, 0.3), p=0.5),
                 # cutout(n_holes=(1, 5), length=(10, 30), p=0.1)
                 ]
        transforms = get_transforms(max_rotate=10, max_zoom=1.2, max_lighting=0.2, do_flip=False,
                                    max_warp=0.10, p_affine=.75, p_lighting=.75, xtra_tfms=extra)
        transforms = list(transforms)
        transforms[1] = []

        df = pd.read_csv(os.path.join(self.hparams.data, 'covid_test_v5.csv'))
        df['Non_Covid'] = 1 - df['Covid']
        if self.hparams.types == 4:
            dset = (
                ImageList.from_df(df=df, path=os.path.join(self.hparams.data, 'data'), cols='Images')
                .split_by_rand_pct(1.0, seed=self.hparams.seed)
                .label_from_df(cols=['Covid', 'Airspace_Opacity', 'Consolidation', 'Pneumonia'], label_cls=MultiCategoryList)
                .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
                .databunch(bs=self.hparams.batch, num_workers=8)
                .normalize(imagenet_stats)
            )
        elif self.hparams.types == 2:
            dset = (
                ImageList.from_df(df=df, path=os.path.join(self.hparams.data, 'data'), cols='Images')
                .split_by_rand_pct(1.0, seed=self.hparams.seed)
                .label_from_df(cols=['Covid', 'Non_Covid'], label_cls=MultiCategoryList)
                .transform(transforms, size=self.hparams.shape, padding_mode='zeros')
                .databunch(bs=self.hparams.batch, num_workers=8)
                .normalize(imagenet_stats)
            )
        return dset.train_dl.dl, dset.valid_dl.dl

    def val_dataloader(self):
        ds_train, ds_valid = self.eval_dataloader()
        return ds_valid

    def custom_step(self, batch, batch_idx, prefix='val'):
        image, target = batch

        output = self.classifier(image)
        loss = torch.nn.BCELoss()(output, target)  # + SoftDiceLoss()(output, target)

        result = OrderedDict({
            f'{prefix}_loss': loss,
            f'{prefix}_output': output,
            f'{prefix}_target': target,
        })
        return result

    def validation_step(self, batch, batch_idx, prefix='val'):
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def test_step(self, batch, batch_idx, prefix='test'):
        return self.custom_step(batch, batch_idx, prefix=prefix)

    def custom_epoch_end(self, outputs, prefix='val'):
        loss_mean = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()

        np_output = torch.cat([x[f'{prefix}_output'].squeeze_(0) for x in outputs], dim=0).to('cpu').numpy()
        np_target = torch.cat([x[f'{prefix}_target'].squeeze_(0) for x in outputs], dim=0).to('cpu').numpy()

        # print(np_output)
        # print(np_target)
        # Casting to binary
        np_output = 1 * (np_output >= self.hparams.threshold).astype(np.uint8)
        np_target = 1 * (np_target >= self.hparams.threshold).astype(np.uint8)

        print(np_target.shape)
        print(np_output.shape)

        result = {}
        result[f'{prefix}_loss'] = loss_mean

        tqdm_dict = {}
        tqdm_dict[f'{prefix}_loss'] = loss_mean

        tb_log = {}
        tb_log[f'{prefix}_loss'] = loss_mean

        f1_scores = []
        np_log = []
        if np_output.shape[0] > 0 and np_target.shape[0] > 0:
            for p in range(self.hparams.types):
                PP = np.sum((np_target[:, p] == 1))
                NN = np.sum((np_target[:, p] == 0))
                TP = np.sum((np_target[:, p] == 1) & (np_output[:, p] == 1))
                TN = np.sum((np_target[:, p] == 0) & (np_output[:, p] == 0))
                FP = np.sum((np_target[:, p] == 0) & (np_output[:, p] == 1))
                FN = np.sum((np_target[:, p] == 1) & (np_output[:, p] == 0))
                np_log.append([p, PP, NN, TP, TN, FP, FN])
                precision_score = (TP / (TP + FP + 1e-12))
                recall_score = (TP / (TP + FN + 1e-12))
                beta = 1
                f1_score = (1 + beta**2) * precision_score * recall_score / \
                    (beta**2 * precision_score + recall_score + 1e-12)
                beta = 2
                f2_score = (1 + beta**2) * precision_score * recall_score / \
                    (beta**2 * precision_score + recall_score + 1e-12)
                # f1_score = sklearn.metrics.fbeta_score(np_target[:, p], np_output[:, p], beta=1, average='macro')
                # f2_score = sklearn.metrics.fbeta_score(np_target[:, p], np_output[:, p], beta=2, average='macro')
                # precision_score = sklearn.metrics.precision_score(np_target[:, p], np_output[:, p], average='macro')
                # recall_score = sklearn.metrics.recall_score(np_target[:, p], np_output[:, p], average='macro')

                f1_scores.append(f1_score)
                tqdm_dict[f'{prefix}_f1_score_{p}'] = f'{f1_score:0.4f}'
                tqdm_dict[f'{prefix}_f2_score_{p}'] = f'{f2_score:0.4f}',
                tqdm_dict[f'{prefix}_precision_score_{p}'] = f'{precision_score:0.4f}'
                tqdm_dict[f'{prefix}_recall_score_{p}'] = f'{recall_score:0.4f}'

                tb_log[f'{prefix}_f1_score_{p}'] = f1_score
                tb_log[f'{prefix}_f2_score_{p}'] = f2_score
                tb_log[f'{prefix}_precision_score_{p}'] = precision_score
                tb_log[f'{prefix}_recall_score_{p}'] = recall_score

            tqdm_dict[f'{prefix}_f1_score_mean'] = f'{np.array(f1_scores).mean():0.4f}'
            tb_log[f'{prefix}_f1_score_mean'] = np.array(f1_scores).mean()
        pprint(np.array(np_log))
        pprint(tqdm_dict)
        result['log'] = tb_log
        np_output = []
        np_target = []
        return result

    def validation_epoch_end(self, outputs, prefix='val'):
        return self.custom_epoch_end(outputs, prefix=prefix)

    def test_epoch_end(self, outputs, prefix='test'):
        return self.custom_epoch_end(outputs, prefix=prefix)


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hparams.load:
        model = MSGGAN(hparams).load_from_checkpoint(hparams.load)
    else:
        model = MSGGAN(hparams)

    custom_log_dir = os.path.join(str(hparams.save),
                                  str(hparams.note),
                                  str(hparams.case),
                                  str(hparams.shape),
                                  str(hparams.types),
                                  ),

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # trainer = Trainer()
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(str(hparams.save),
                              str(hparams.note),
                              str(hparams.case),
                              str(hparams.Lambda),
                              str(hparams.pathology),
                              str(hparams.shape),
                              str(hparams.types),
                              # str(hparams.folds),
                              # str(hparams.valid_fold_index),
                              'ckpt'),
        save_top_k=hparams.epochs,  # 10,
        verbose=True,
        monitor='val_f1_score_mean' if hparams.pathology == 'All' else 'val_f1_score_0',  # TODO
        mode='max'
    )

    trainer = Trainer(
        num_sanity_val_steps=0,
        default_root_dir=os.path.join(str(hparams.save),
                                      str(hparams.note),
                                      str(hparams.case),
                                      str(hparams.Lambda),
                                      str(hparams.pathology),
                                      str(hparams.shape),
                                      str(hparams.types),
                                      ),
        default_save_path=os.path.join(str(hparams.save),
                                       str(hparams.note),
                                       str(hparams.case),
                                       str(hparams.Lambda),
                                       str(hparams.pathology),
                                       str(hparams.shape),
                                       str(hparams.types),
                                       ),
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        checkpoint_callback=checkpoint_callback,
        progress_bar_refresh_rate=1,
        early_stop_callback=None,
        fast_dev_run=hparams.fast_dev_run,
        # train_percent_check=hparams.percent_check,
        # val_percent_check=hparams.percent_check,
        # test_percent_check=hparams.percent_check,
        # distributed_backend=hparams.distributed_backend,
        # use_amp=hparams.use_16bit,
        val_check_interval=hparams.percent_check,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # trainer.fit(model)
    if hparams.eval:
        assert hparams.loadD
        model.eval()
        # trainer.test(model)
        pass
    elif hparams.pred:
        assert hparams.load
        model.eval()
        pass
    else:
        trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--b1", type=float, default=0.0)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--latent_dim", type=int, default=16)

    # hparams = parser.parse_args()
    # parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--latent_size", type=int, default=512)

    parser.add_argument('--data', metavar='DIR', default=".", type=str)
    parser.add_argument('--save', metavar='DIR', default="train_log", type=str)
    parser.add_argument('--info', metavar='DIR', default="train_log")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2020)
    # Regular, warmup, pretrained
    parser.add_argument('--note', type=str, default="MSGGAN")
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
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--types', type=int, default=4)
    parser.add_argument('--shape', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pathology', default='All')
    hparams = parser.parse_args()

    main(hparams)
