from enum import Enum

import torchvision
import torch


class Augmentation(Enum):
    ORIGINAL = 'ORIGINAL'
    HORIZONTAL_FLIP = 'HORIZONTAL-FLIP'
    BLUR = 'BLUR'
    CONTRAST = 'CONTRAST'
    BRIGHTNESS = 'BRIGHTNESS'
    HUE = 'HUE'
    ROTATION = 'ROTATION'

class FunctionsUtils:

    @staticmethod
    def apply_gaussian_blur(data, kernel_size):
        return torchvision.transforms.functional.gaussian_blur(img=data, kernel_size=kernel_size)

    @staticmethod
    def apply_horizontal_flip(x: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.functional.hflip(img=x)

    @staticmethod
    def apply_contrast(data: torch.Tensor, contrast_factor: float) -> torch.Tensor:
        return torchvision.transforms.functional.adjust_contrast(
            img=data,
            contrast_factor=contrast_factor,
        )

    @staticmethod
    def apply_brightness(self, data: torch.Tensor, brightness_factor: float) -> torch.Tensor:
        return torchvision.transforms.functional.adjust_brightness(
            img=data,
            brightness_factor=brightness_factor,
        )

    # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_hue.html#torchvision.transforms.functional.adjust_hue
    # https://en.wikipedia.org/wiki/Hue
    @staticmethod
    def apply_hue(data: torch.Tensor, hue_factor: float) -> torch.Tensor:
        '''
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    '''
        if hue_factor < -0.5 or hue_factor > 0.5:
            raise ValueError('hue_factor must be in [-0.5, 0.5]')
        return torchvision.transforms.functional.adjust_hue(
            img=data,
            hue_factor=hue_factor,
        )

    @staticmethod
    def apply_rotation(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        return torchvision.transforms.functional.affine(img=x, translate=(0, 0), angle=angle_deg, shear=0.0, scale=1.0)
