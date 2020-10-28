from argparse import Namespace

import torch.nn as nn

from byol import BYOL, default_augmentation
from cifar10_datamodule import CIFAR10DataModule
from train_utils import byol_main, vit_main
from vit import VisionTransformer


class CIFAR10VisionTransformer(VisionTransformer):
    def __init__(self, *args,
                 dim=None,
                 patch_size=None,
                 n_patches=None,
                 n_layers=None,
                 n_heads=None,
                 dropout=None,
                 **kwargs):
        super().__init__(*args,
                         dim=dim,
                         patch_size=patch_size,
                         n_patches=n_patches,
                         n_layers=n_layers,
                         n_heads=n_heads,
                         dropout=dropout,
                         **kwargs)
        n_classes = 10
        self.classifier = nn.Linear(dim, n_classes)


vit_args = {
    "n_channels": 3,
    "patch_size": 8,
    "dim": 256,
    "n_patches": (32//8)**2,
    "n_layers": 8,
    "n_heads": 4,
    "dropout": 0,
}


class CIFAR10BYOL(BYOL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = default_augmentation(32, is_color_image=True)
        vit = CIFAR10VisionTransformer.from_argparse_args(
            Namespace(), **vit_args)
        vit.classifier = nn.Identity()
        self.online_encoder = vit
        self.copy()


byol_args = {
    "projector_isize": 256,
    "projector_hsize": 256,
    "projector_osize": 128,
    "predictor_hsize": 256
}


def vit_weights_from_checkpoint(checkpoint_path):
    vit = CIFAR10BYOL.load_from_checkpoint(checkpoint_path).online_encoder
    vit.classifier = nn.Linear(256, 10)
    return vit.state_dict()


if __name__ == '__main__':
    best_checkpoint_path = \
        byol_main(CIFAR10DataModule, CIFAR10BYOL, byol_args, "byol_cifar10")

    best_weights = vit_weights_from_checkpoint(best_checkpoint_path)

    vit_main(CIFAR10DataModule, CIFAR10VisionTransformer, vit_args,
             "vit_cifar10", best_weights)
