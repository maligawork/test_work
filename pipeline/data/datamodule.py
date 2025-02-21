from typing import Union

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .dataset import TriangleSquareDataset
from .augmentation import DataAugmentation

class TriangleSquareDataModule(pl.LightningDataModule):
    """Lightning data module.
        Params:
            cfg (DictConfig): read more in configs-folder.
            batch_size (int): batch size.
            num_workers (int): number of workers.
            device (str): device to use.
    """
    def __init__(self,
                 cfg,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 device: Union[str, torch.device] = 'cpu',
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def setup(self, stage: str):
        """Method to init datasets."""
        self.transforms = DataAugmentation(self.cfg.p_rotation) # поворот в низком разрешении превращает квадрат в круг

        if stage == 'fit':
            self.train_dataset = TriangleSquareDataset(num_samples=self.cfg.num_samples, image_size=self.cfg.image_size, 
                                                       max_size=self.cfg.max_size, device=self.device, transforms=self.transforms)
            self.val_dataset = TriangleSquareDataset(num_samples=self.cfg.num_samples, image_size=self.cfg.image_size, 
                                                     max_size=self.cfg.max_size, device=self.device, transforms=self.transforms)

        if stage == 'test':
            self.test_dataset = TriangleSquareDataset(num_samples=self.cfg.num_samples, image_size=self.cfg.image_size, 
                                                      max_size=self.cfg.max_size, device=self.device, transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
        )

    # @staticmethod
    # def collate_fn(batch):
    #     """ To handle the data loading as different images may have different number
    #         of objects and to handle varying size tensors as well.
    #     """
    #     return tuple(list(map(list, zip(*batch))))
            