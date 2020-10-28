import torch
import torch.nn as nn

from byol import BYOL, default_augmentation
from stl10_datamodule import STL10DataModule
from train_utils import main


class BYOLSTL10(BYOL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = default_augmentation(96, is_color_image=True)
        resnet18 = torch.hub.load(
            'pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        resnet18.fc = nn.Identity()
        self.online_encoder = resnet18
        self.copy()


byol_args = {
    "projector_isize": 512,
    "projector_hsize": 512,
    "projector_osize": 128,
    "predictor_hsize": 512
}


if __name__ == '__main__':
    main(STL10DataModule, BYOLSTL10, byol_args, "byol_stl10")
