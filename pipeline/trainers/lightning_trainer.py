import os

import torch
from torch.optim import lr_scheduler
import pytorch_lightning as pl

# from pipeline.utils.schedulers import cosine_warmup_scheduler
from ..utils.visualize import draw_results


class LightningTrainerSiamese(pl.LightningModule):
    """Main class for siamese models lightning module.
        Params:
            model (torch.nn.Model): torch neural network model.
            loss_fn (func): loss function.
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer_params,
                 lr_scheduler_params,
                 metrics_threshold,
                 *args,
                 **kwargs,
        ):
        super().__init__()

        # Model
        self.model = model
        self.loss_fn = loss_fn

        # Optimizer params
        self.optimizer_params = optimizer_params

        # scheduler params
        self.lr_scheduler_params = lr_scheduler_params

        # Metrics params
        self.train_results_save_dir = None
        self.val_results_save_dir = None
        self.test_results_save_dir = None

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.metrics_threshold = metrics_threshold

        self.save_hyperparameters(ignore=['model', 'loss_fn'])   # model checkpointing separately
        self.devices_count = 1

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.optimizer_params['optimizer_type'] == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.optimizer_params['lr0'],
                betas=(self.optimizer_params['momentum'], 0.999),
                weight_decay=self.optimizer_params['weight_decay'],
            )
        elif self.optimizer_params['optimizer_type'] == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.optimizer_params['lr0'],
                momentum=self.optimizer_params['momentum'], 
                nesterov=self.optimizer_params['nesterov'],
                weight_decay=self.optimizer_params['weight_decay'],
            )
        else:
            raise TypeError

        if self.lr_scheduler_params['scheduler_type'] == 'cosine_warmup': 
            pass
            # scheduler = cosine_warmup_scheduler(optimizer, self.lr0, kwargs.lrf, kwargs.lrw, 
            #                                        kwargs.warmup_epochs, self.trainer.max_epochs)
        elif self.lr_scheduler_params['scheduler_type'] == 'cosine_annealing':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.lr_scheduler_params['t_max'], 
                                                       eta_min=self.lr_scheduler_params['eta_min'])
        else:
            raise TypeError

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self):
        """Start training."""
        self.train_results_save_dir = os.path.join(self.trainer.logger.save_dir, 'results', 'train_results', str(self.global_rank))
        if not os.path.exists(self.train_results_save_dir):
            os.makedirs(self.train_results_save_dir)
        self.devices_count = len(self.trainer.device_ids)
        
    def on_validation_start(self):
        """Start validation."""
        self.val_results_save_dir = os.path.join(self.trainer.logger.save_dir, 'results', 'val_results', str(self.global_rank))
        if not os.path.exists(self.val_results_save_dir):
            os.makedirs(self.val_results_save_dir)
            
    def on_test_start(self):
        """Start test."""
        self.test_results_save_dir = os.path.join(self.trainer.logger.save_dir, 'results','test_results', str(self.global_rank))
        if not os.path.exists(self.test_results_save_dir):
            os.makedirs(self.test_results_save_dir)

    def on_train_batch_end(self, out, batch, batch_idx):
        """Calculation and log train results."""
        if batch_idx == (len(self.trainer.datamodule.train_dataset) // batch[0][0].shape[0]) - 1:
            (imgs1, imgs2), labels = batch
            draw_results(imgs1, imgs2, labels, out['pred'], self.train_results_save_dir, n=3)
        del out['pred']
        self.training_step_outputs.append(out)

    def on_validation_batch_end(self, out, batch, batch_idx):
        """Calculation and log validation results."""
        if batch_idx == (len(self.trainer.datamodule.val_dataset) // batch[0][0].shape[0]) - 1:
            (imgs1, imgs2), labels = batch
            draw_results(imgs1, imgs2, labels, out['pred'], self.val_results_save_dir, n=3)
        del out['pred']
        self.validation_step_outputs.append(out)
    
    def on_test_batch_end(self, out, batch, batch_idx):
        """Calculation and log test results."""
        if batch_idx == (len(self.trainer.datamodule.test_dataset) // batch[0][0].shape[0]) - 1:
            (imgs1, imgs2), labels = batch
            draw_results(imgs1, imgs2, labels, out['pred'], self.test_results_save_dir, n=7)
        del out['pred']
        self.test_step_outputs.append(out)