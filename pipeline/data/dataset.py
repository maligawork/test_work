import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Union


class TriangleSquareDataset(Dataset):
    def __init__(self, 
                 num_samples: int = 1000, 
                 image_size: int = 32, 
                 max_size: int = 15, 
                 device: Union[str, torch.device] = 'cpu',
                 transforms: nn.Module = None,
        ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_size = max_size
        self.device = device
        self.transforms = transforms

        assert self.max_size < self.image_size // 2 # чтобы фигуры не выходили за границы изображения
        assert self.max_size > 4 # чтобы фигуры были больше 4 пикселей (чтобы было видно круг)
        
    def _generate_random_shape(self):
        """Generate single image with random shape."""
        img = torch.zeros((3, self.image_size, self.image_size)).to(self.device)
        
        # Случайный выбор формы (0 - круг, 1 - квадрат)
        shape_type = torch.randint(0, 2, (1,)).item()
        
        # Случайная позиция центра
        center_x = torch.randint(self.max_size, self.image_size-self.max_size, (1,)).item()
        center_y = torch.randint(self.max_size, self.image_size-self.max_size, (1,)).item()
        
        if shape_type == 0:  # Круг
            # Случайный размер (от 4 до max_size (радиус круга))
            size = torch.randint(4, self.max_size, (1,)).item()
            radius_squared = size * size
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if (i - center_y)**2 + (j - center_x)**2 < radius_squared:
                        img[:, i, j] = 1.0
        else:  # Квадрат
            # Случайный размер (от 2 до max_size (сторона квадрата))
            size = torch.randint(2, self.max_size, (1,)).item()

            x1, x2 = center_x-size, center_x+size
            y1, y2 = center_y-size, center_y+size
            img[:, y1:y2, x1:x2] = 1.0
            
        return img, shape_type

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Генерируем пару изображений
        img1, shape1 = self._generate_random_shape()
        img2, shape2 = self._generate_random_shape()

        if self.transforms:
            img1 = self.transforms(img1[None, ...])[0]
            img2 = self.transforms(img2[None, ...])[0]
        # Формируем лейбл (1 если формы одинаковые, 0 если разные)
        label = torch.tensor(1.0 if shape1 == shape2 else 0.0, dtype=torch.float32, device=self.device).view(-1)

        return (img1, img2), label
