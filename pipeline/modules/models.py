import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, 
                 encoder_name: str = 'resnet18',
                 pretrained: bool = True,
        ):
        super().__init__()
        self.encoder_name = encoder_name
        self.pretrained = pretrained

        self.backbone = self.define_backbone()
        
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # Изменяем первый слой для работы с 1 каналом вместо 3
        self.backbone.fc = nn.Identity()  # Убираем последний слой, чтобы получить эмбеддинги

    def define_backbone(self):
        if self.encoder_name == 'resnet18':
            return models.resnet18(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_name}")

    def forward(self, x):
        return self.backbone(x)  # Получаем эмбеддинг

# Сиамская сеть
class SiameseNetwork(nn.Module):
    def __init__(self, 
                 encoder_name: str = 'resnet18',
                 pretrained: bool = True,
        ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(encoder_name, pretrained)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)  # Извлекаем признаки
        feat2 = self.feature_extractor(img2)

        diff = torch.abs(feat1 - feat2)  # Вычисляем разницу эмбеддингов

        x = torch.relu(self.fc1(diff))
        x = self.fc2(x)
        return x