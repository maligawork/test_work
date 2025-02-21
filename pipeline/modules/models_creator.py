import torch
from .basic_model_creator import BasicModelCreator
from .models import SiameseNetwork

class SiameseModelsCreator(BasicModelCreator):
    """The general creator for all models.
        Params:
            encoder (str): type of model architecture encoder.
            pretrained (bool): use pretrained model.
    """
    def __init__(self,
                 *args,
                 **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.model = self.create_model()
        self.loss_fn = self.create_loss_fn()

    def create_model(self):
        return SiameseNetwork(
            encoder_name=self.encoder,
            pretrained=self.pretrained,
        )

    def create_loss_fn(self):
        loss_name = self.losses['loss']
        if loss_name == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise TypeError

    def load_model_from_checkpoint(self, checkpoint_path: str, device: str = 'cpu'):
        pass