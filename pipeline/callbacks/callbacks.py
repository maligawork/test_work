import pytorch_lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor


class _LearningRateMonitor(LearningRateMonitor, pl.Callback):
    """For version compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)