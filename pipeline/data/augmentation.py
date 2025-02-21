from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import kornia.augmentation as kA
from kornia.core import Tensor
from kornia.constants import Resample


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors.
        Params:
            p_rotation (float): probability of the image being rotated.
    """
    def __init__(
            self,
            p_rotation: float = 0.5,
        ):
        super().__init__()
        self.rotation = kA.RandomRotation(degrees=180, p=p_rotation)
        self.transforms = [self.rotation]

    @torch.no_grad()  # disable gradients for efficiency
    def forward(
        self,
        batch_images: Tensor,
    ):
        """Apply transform.
            Args:
                batch_images (torch.Tensor): batch of tensor images.
            Returns:
                batch_images (torch.Tensor): transformed batch of tensor images.
        """

        for transform in self.transforms:
            batch_images = transform(batch_images)

        return batch_images
