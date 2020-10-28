import copy
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import (ColorJitter, RandomGrayscale,
                                 RandomHorizontalFlip, RandomResizedCrop,
                                 RandomSolarize)
from kornia.filters import GaussianBlur2d
from pl_bolts.optimizers import LARSWrapper, LinearWarmupCosineAnnealingLR
from sklearn.linear_model import LogisticRegression
from torch.optim import SGD

from argparse_utils import from_argparse_args


class RandomApply(nn.Module):
    def __init__(self, proc, p):
        super().__init__()

        self.proc = proc
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x

        return self.proc(x)


def default_augmentation(image_size, is_color_image=True):
    def get_kernel_size(image_size):
        kernel_size = image_size // 10
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel_size = (kernel_size, kernel_size)

        return kernel_size

    def color_model(gaussian_p, solarize_p):
        return nn.Sequential(
            RandomResizedCrop((image_size, image_size),
                              interpolation="BICUBIC"),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            RandomGrayscale(p=0.2),
            RandomApply(GaussianBlur2d(get_kernel_size(
                image_size), (0.1, 2.0)), p=gaussian_p),
            RandomSolarize(0, 0, p=solarize_p)
        )

    def monochrome_model(gaussian_p, solarize_p):
        return nn.Sequential(
            RandomResizedCrop((image_size, image_size),
                              interpolation="BICUBIC"),
            RandomHorizontalFlip(),
            RandomApply(GaussianBlur2d(get_kernel_size(
                image_size), (0.1, 2.0)), p=gaussian_p),
            RandomSolarize(0, 0, p=solarize_p)
        )

    if is_color_image:
        return [color_model(*ps) for ps in [(1., 0.), (0.1, 0.2)]]
    else:
        return [monochrome_model(*ps) for ps in [(1., 0.), (0.1, 0.2)]]


class TargetNetworkUpdator(pl.Callback):
    def __init__(self, tau=0.996):
        super().__init__()

        self.tau = tau

    def on_train_batch_end(self, trainer, pl_module,
                           batch, batch_idx, dataloader_idx):
        def update(src, dst):
            for sp, dp in zip(src.parameters(), dst.parameters()):
                dp.data = sp.data * \
                    (1 - self.tau) + dp.data * self.tau

        update(pl_module.online_encoder, pl_module.target_encoder)
        update(pl_module.online_projector, pl_module.target_projector)


class PredictorInitializer(pl.Callback):
    def __init__(self, step=10, max_value=0.01):
        super().__init__()

        self.step = step
        self.max_value = max_value

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.step == 0:
            for p in pl_module.predictor.parameters():
                p.data = torch.rand_like(p.data) * self.max_value


class MLP(nn.Module):
    def __init__(self, isize, hsize, osize):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(isize, hsize, bias=False),
            nn.BatchNorm1d(hsize),
            nn.ReLU(inplace=True),
            nn.Linear(hsize, osize)
        )

    def forward(self, x):
        return self.net(x)


class BYOL(pl.LightningModule):
    def __init__(self,
                 *args,
                 projector_isize=None,
                 projector_hsize=None,
                 projector_osize=None,
                 predictor_hsize=None,
                 learning_rate=None,
                 weight_decay=None,
                 max_epochs=None,
                 warmup_epochs=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.online_projector = MLP(
            projector_isize, projector_hsize, projector_osize)
        self.predictor = MLP(projector_osize, predictor_hsize, projector_osize)

    def copy(self):
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

    def forward(self, x):
        return self.online_encoder(x)

    def calc_loss(self, x):
        def proc(x, y):
            out_online = self.predictor(
                self.online_projector(self.online_encoder(x)))

            with torch.no_grad():
                out_target = self.target_projector(self.target_encoder(y))

            return -2 * F.cosine_similarity(out_online,
                                            out_target.detach(),
                                            dim=-1).mean()

        if isinstance(self.augment, nn.Module):
            imgs = [self.augment(x), self.augment(x)]
        elif isinstance(self.augment, (list, tuple)):
            imgs = [self.augment[0](x), self.augment[1](x)]
        else:
            raise TypeError("augmentation must be torch Module or its list")

        return proc(imgs[0], imgs[1]) + proc(imgs[1], imgs[0])

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.calc_loss(x)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, label = batch

        with torch.no_grad():
            y = self.online_encoder(x)

        return {'representation': y, 'label': label}

    def validation_epoch_end(self, outputs):
        ys = torch.cat([o['representation'] for o in outputs]).cpu().numpy()
        labels = torch.cat([o['label'] for o in outputs]).cpu().numpy()
        half_idx = len(ys) // 2

        classifier = LogisticRegression(max_iter=100, solver="liblinear")

        classifier.fit(ys[:half_idx], labels[:half_idx])
        val_acc = 100 * classifier.score(ys[half_idx:], labels[half_idx:])

        logs = {'val_acc': torch.tensor(val_acc)}
        results = {'log': logs}

        return results

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.hparams.learning_rate,
                        momentum=0,
                        weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, self.hparams.warmup_epochs, self.hparams.max_epochs,
            warmup_start_lr=1e-2, eta_min=1e-2)

        return [optimizer], [scheduler]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.2)
        parser.add_argument('--weight_decay', type=float, default=1e-6)
        parser.add_argument('--warmup_epochs', type=int, default=10)

        return parser
