import abc


class BasicModelCreator(metaclass=abc.ABCMeta):
    """Main class for models.
        Params:
            encoder (str): type of model architecture encoder.
            pretrained (bool): use pretrained model.
    """
    def __init__(self,
                 encoder: str,
                 pretrained: bool = False,
                 losses: dict = None,
                 **kwargs,
        ):
        self.encoder = encoder
        self.pretrained = pretrained
        self.losses = losses

    @abc.abstractmethod
    def create_model(self):
        """Define torch model."""
        pass

    @abc.abstractmethod
    def create_loss_fn(self):
        """Define loss function."""
        pass

    @abc.abstractmethod
    def load_model_from_checkpoint(self, checkpoint_path: str, device: str = 'cpu'):
        """Load model from checkpoint."""
        pass